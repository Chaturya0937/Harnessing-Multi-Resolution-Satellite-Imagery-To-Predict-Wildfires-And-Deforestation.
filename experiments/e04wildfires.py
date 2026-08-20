import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc

# 1. Device Selection (CUDA GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
if torch.cuda.is_available():
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

# 2. Dataset Loader with Normalization & NaN Cleaning
class EO4WildFiresDataset(Dataset):
    def __init__(self, data_dir, target_size=(256, 256)):
        self.file_paths = glob.glob(os.path.join(data_dir, "**/*.nc"), recursive=True)
        if len(self.file_paths) == 0:
            self.file_paths = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        if len(self.file_paths) == 0:
            self.file_paths = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
            
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        if file_path.endswith('.nc'):
            with nc.Dataset(file_path, 'r') as ds:
                inputs = np.array(ds.variables['S2A'][:], dtype=np.float32)
                mask = np.array(ds.variables['burned_mask'][:], dtype=np.float32)
        elif file_path.endswith('.npz'):
            data = np.load(file_path)
            inputs = data["S2A"].astype(np.float32)
            mask = data["burned_mask"].astype(np.float32)
        else:
            data = np.load(file_path)
            inputs = data[:6, :, :].astype(np.float32)
            mask = data[-1, :, :].astype(np.float32)

        # Clean NaN/Inf values
        inputs = np.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale Sentinel-2 reflectance (0 - 10000 range -> 0.0 - 1.0)
        inputs = np.clip(inputs, 0.0, 10000.0) / 10000.0
        mask = np.clip(mask, 0.0, 1.0)

        inputs_tensor = torch.from_numpy(inputs)
        mask_tensor = torch.from_numpy(mask)
        
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # Spatial resizing
        if inputs_tensor.shape[1:] != self.target_size:
            inputs_tensor = F.interpolate(
                inputs_tensor.unsqueeze(0), 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

        if mask_tensor.shape[1:] != self.target_size:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0), 
                size=self.target_size, 
                mode='nearest'
            ).squeeze(0)

        return inputs_tensor, mask_tensor

# 3. Hybrid Loss Function (BCE + Dice Loss)
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return bce_loss + dice_loss

# 4. Evaluation Metrics
def calculate_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum().item()
    total_pred = preds_flat.sum().item()
    total_target = targets_flat.sum().item()
    union = total_pred + total_target - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (total_pred + total_target + smooth)
    
    return iou, dice

# 5. Model Architecture
class LightUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        d1 = torch.cat([d1, e1], dim=1)
        return self.dec1(d1)

# 6. Main Execution Pipeline
def main():
    search_dir = r"C:\Users\Vardhan\Desktop\Coding\Minor Project\experiments\eo4wildfires_local\extracted\eo4wildfires"
    dataset = EO4WildFiresDataset(search_dir, target_size=(256, 256))

    if len(dataset) == 0:
        raise ValueError(f"No files found in '{search_dir}'.")

    print(f"Successfully loaded {len(dataset)} samples. Preparing DataLoader...")

    sample_input, _ = dataset[0]
    in_channels = sample_input.shape[0]
    print(f"Detected input channels: {in_channels}")

    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )

    model = LightUNet(in_channels=in_channels, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    epochs = 5
    model.train()
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        
        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            iou, dice = calculate_metrics(outputs.detach(), targets)
            
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step + 1}/{len(dataloader)}] "
                    f"| Loss: {loss.item():.4f} | IoU: {iou:.4f} | Dice (F1): {dice:.4f}"
                )

        avg_loss = running_loss / len(dataloader)
        avg_iou = running_iou / len(dataloader)
        avg_dice = running_dice / len(dataloader)
        
        print(
            f"\n=== Epoch {epoch} Summary ===\n"
            f"Avg Loss: {avg_loss:.4f} | Avg IoU: {avg_iou:.4f} | Avg Dice Score: {avg_dice:.4f}\n"
            f"=========================\n"
        )

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/wildfire_severity_unet.pth")
    print("Clean model saved to './models/wildfire_severity_unet.pth'")

if __name__ == "__main__":
    main()