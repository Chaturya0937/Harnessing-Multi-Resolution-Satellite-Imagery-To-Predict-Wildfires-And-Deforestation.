"""
Train a multi-class WILDFIRE SEVERITY segmentation model on the GeoTIFFs
produced by fetch_severity_data.py.

Each GeoTIFF has 13 bands: 6 pre-fire + 6 post-fire Sentinel-2 bands, plus
1 severity label band (0=unburned, 1=low, 2=moderate-low, 3=moderate-high,
4=high). The model takes the 12 image bands as input and predicts a
severity class per pixel (instead of your earlier binary burned/not-burned
model).

Architecture choice: same as your binary model - a pretrained U-Net via
segmentation_models_pytorch with an EfficientNet-B0 encoder - this remains
the practical, laptop-friendly, well-supported choice. The only real
changes from the binary version are: 12 input channels instead of 6,
5 output classes instead of 1, and CrossEntropyLoss instead of BCE+Dice.

Install:
    pip install segmentation-models-pytorch rasterio tqdm matplotlib
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError("pip install segmentation-models-pytorch") from e
try:
    from tqdm import tqdm
except ImportError as e:
    raise ImportError("pip install tqdm") from e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "./severity_tiffs"       # folder with GeoTIFFs downloaded from Drive
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4                       # 12 channels + 5-class head uses more VRAM
                                      # than the binary model; drop to 2 if you hit OOM
VAL_FRACTION = 0.15
EPOCHS = 60
EARLY_STOP_PATIENCE = 10
ENCODER_NAME = "efficientnet-b0"
LEARNING_RATE = 3e-4
NUM_WORKERS = 0
SEED = 42
NUM_CLASSES = 5
CLASS_NAMES = ["unburned", "low", "moderate-low", "moderate-high", "high"]
MODEL_OUT_DIR = "./models"
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "wildfire_severity_multiclass_best.pth")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}", flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SeverityDataset(Dataset):
    def __init__(self, data_dir, target_size=(256, 256), augment=False):
        self.file_paths = glob.glob(os.path.join(data_dir, "**/*.tif"), recursive=True)
        self.file_paths += glob.glob(os.path.join(data_dir, "**/*.tiff"), recursive=True)
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with rasterio.open(file_path) as src:
            arr = src.read().astype(np.float32)  # shape: (13, H, W)

        images = arr[:12]          # pre(6) + post(6) bands
        label = arr[12]            # severity class band

        images = np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
        # Sentinel-2 SR reflectance scaling (0-10000 -> 0-1)
        images = np.clip(images, 0.0, 10000.0) / 10000.0

        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.clip(label, 0, NUM_CLASSES - 1).astype(np.int64)

        images_tensor = torch.from_numpy(images)
        label_tensor = torch.from_numpy(label).unsqueeze(0).float()  # for interpolate

        if images_tensor.shape[1:] != self.target_size:
            images_tensor = F.interpolate(
                images_tensor.unsqueeze(0), size=self.target_size,
                mode="bilinear", align_corners=False
            ).squeeze(0)
        if label_tensor.shape[1:] != self.target_size:
            label_tensor = F.interpolate(
                label_tensor.unsqueeze(0), size=self.target_size, mode="nearest"
            ).squeeze(0)

        label_tensor = label_tensor.squeeze(0).long()  # back to (H, W) int64 for CrossEntropy

        if self.augment:
            images_tensor, label_tensor = self._augment(images_tensor, label_tensor)

        return images_tensor, label_tensor

    @staticmethod
    def _augment(images_tensor, label_tensor):
        if random.random() < 0.5:
            images_tensor = torch.flip(images_tensor, dims=[2])
            label_tensor = torch.flip(label_tensor, dims=[1])
        if random.random() < 0.5:
            images_tensor = torch.flip(images_tensor, dims=[1])
            label_tensor = torch.flip(label_tensor, dims=[0])
        k = random.randint(0, 3)
        if k > 0:
            images_tensor = torch.rot90(images_tensor, k, dims=[1, 2])
            label_tensor = torch.rot90(label_tensor, k, dims=[0, 1])
        return images_tensor, label_tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(in_channels=12, num_classes=NUM_CLASSES, encoder_name=ENCODER_NAME):
    print(f"Building {encoder_name} U-Net (multiclass, {num_classes} classes)...", flush=True)
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    print("Model built.", flush=True)
    return model


# ---------------------------------------------------------------------------
# Metrics: per-class IoU + mean IoU (mIoU) - the standard metric for
# multi-class segmentation
# ---------------------------------------------------------------------------
def per_class_iou(preds, targets, num_classes, smooth=1e-6):
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum().item()
        union = pred_c.sum().item() + target_c.sum().item() - intersection
        ious.append((intersection + smooth) / (union + smooth))
    return ious


# ---------------------------------------------------------------------------
# Train / val
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, scaler, train=True, desc=""):
    model.train() if train else model.eval()
    running_loss = 0.0
    class_iou_sums = np.zeros(NUM_CLASSES)
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    pbar = tqdm(loader, desc=desc, leave=False)
    with context:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            preds = torch.argmax(logits.detach(), dim=1)
            ious = per_class_iou(preds, labels, NUM_CLASSES)
            class_iou_sums += np.array(ious)
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", mIoU=f"{np.mean(ious):.4f}")

    return running_loss / n_batches, class_iou_sums / n_batches


def main():
    full_dataset = SeverityDataset(DATA_DIR, target_size=IMAGE_SIZE, augment=False)
    if len(full_dataset) == 0:
        raise ValueError(f"No .tif files found in '{DATA_DIR}'.")
    print(f"Found {len(full_dataset)} tiles.", flush=True)

    val_size = max(1, int(len(full_dataset) * VAL_FRACTION))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_dataset = SeverityDataset(DATA_DIR, target_size=IMAGE_SIZE, augment=True)
    train_dataset.file_paths = [full_dataset.file_paths[i] for i in train_subset.indices]
    val_dataset = SeverityDataset(DATA_DIR, target_size=IMAGE_SIZE, augment=False)
    val_dataset.file_paths = [full_dataset.file_paths[i] for i in val_subset.indices]

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    best_miou = -1.0
    epochs_without_improvement = 0

    print("Starting training loop...", flush=True)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_class_ious = run_epoch(
            model, train_loader, criterion, optimizer, scaler, train=True,
            desc=f"Epoch {epoch}/{EPOCHS} [train]"
        )
        val_loss, val_class_ious = run_epoch(
            model, val_loader, criterion, optimizer, scaler, train=False,
            desc=f"Epoch {epoch}/{EPOCHS} [val]"
        )

        train_miou = train_class_ious.mean()
        val_miou = val_class_ious.mean()
        scheduler.step(val_miou)

        print(f"\n=== Epoch {epoch}/{EPOCHS} ===", flush=True)
        print(f"Train | Loss: {train_loss:.4f} | mIoU: {train_miou:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | mIoU: {val_miou:.4f}")
        for name, iou in zip(CLASS_NAMES, val_class_ious):
            print(f"  val IoU [{name}]: {iou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_OUT_PATH)
            print(f"  -> New best val mIoU ({best_miou:.4f}). Saved to {MODEL_OUT_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print("No improvement - stopping early.")
                break

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
