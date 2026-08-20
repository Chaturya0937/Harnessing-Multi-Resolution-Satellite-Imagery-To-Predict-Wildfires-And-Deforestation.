import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on device: {device}")

# 1. Preprocessed Dataset Loader
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

        # Preprocessing: Clean NaNs and Normalize
        inputs = np.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        inputs = np.clip(inputs, 0.0, 10000.0) / 10000.0
        mask = np.clip(mask, 0.0, 1.0)

        inputs_tensor = torch.from_numpy(inputs)
        mask_tensor = torch.from_numpy(mask)
        
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        if inputs_tensor.shape[1:] != self.target_size:
            inputs_tensor = F.interpolate(inputs_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)

        if mask_tensor.shape[1:] != self.target_size:
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)

        return inputs_tensor, mask_tensor

# 2. U-Net Model Architecture
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

# 3. Main Metric Evaluation Routine
def evaluate_model(max_samples=2000):
    model_path = "./models/wildfire_severity_unet.pth"
    search_dir = r"C:\Users\Vardhan\Desktop\Coding\Minor Project\experiments\eo4wildfires_local\extracted\eo4wildfires"

    dataset = EO4WildFiresDataset(search_dir)
    if len(dataset) == 0:
        print("No dataset files found.")
        return

    # Dynamic Channel Matching
    sample_input, _ = dataset[0]
    in_channels = sample_input.shape[0]

    model = LightUNet(in_channels=in_channels, out_channels=1).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint file not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded trained model weights from '{model_path}'")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    total_samples = min(len(dataset), max_samples)
    print(f"Evaluating model performance across {total_samples} test samples...")

    all_iou, all_dice, all_precision, all_recall = [], [], [], []
    processed_count = 0

    vis_inputs, vis_gt, vis_preds = [], [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            if processed_count >= total_samples:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Store first batch samples for plotting visual grid
            if len(vis_inputs) < 3:
                for b in range(min(3, inputs.shape[0])):
                    vis_inputs.append(inputs[b].cpu().numpy())
                    vis_gt.append(targets[b].squeeze().cpu().numpy())
                    vis_preds.append(preds[b].squeeze().cpu().numpy())

            # Per-sample metric calculations
            for b in range(inputs.shape[0]):
                if processed_count >= total_samples:
                    break

                p = preds[b].view(-1)
                t = targets[b].view(-1)

                tp = (p * t).sum().item()
                fp = (p * (1 - t)).sum().item()
                fn = ((1 - p) * t).sum().item()

                smooth = 1e-6
                iou = (tp + smooth) / (tp + fp + fn + smooth)
                dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
                precision = (tp + smooth) / (tp + fp + smooth)
                recall = (tp + smooth) / (tp + fn + smooth)

                all_iou.append(iou)
                all_dice.append(dice)
                all_precision.append(precision)
                all_recall.append(recall)

                processed_count += 1

    mean_iou = np.mean(all_iou)
    mean_dice = np.mean(all_dice)
    mean_precision = np.mean(all_precision)
    mean_recall = np.mean(all_recall)

    print("\n" + "="*45)
    print(f" EVALUATION SUMMARY ({processed_count} SAMPLES)")
    print("="*45)
    print(f" Mean IoU (Intersection over Union) : {mean_iou:.4f}")
    print(f" Mean Dice Score (F1 Score)        : {mean_dice:.4f}")
    print(f" Mean Precision                    : {mean_precision:.4f}")
    print(f" Mean Recall                       : {mean_recall:.4f}")
    print("="*45)

    plot_metrics_and_samples(vis_gt, vis_preds, mean_iou, mean_dice, mean_precision, mean_recall, all_iou)

# 4. Visualization Plotting Function
def plot_metrics_and_samples(vis_gt, vis_preds, m_iou, m_dice, m_prec, m_rec, all_iou):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Visual Predictions vs Ground Truth
    for i in range(min(3, len(vis_gt))):
        axes[0, i].imshow(vis_gt[i], cmap='gray')
        axes[0, i].set_title(f"Sample {i+1}: Ground Truth")
        axes[0, i].axis('off')

    for i in range(min(3, len(vis_preds))):
        axes[1, i].imshow(vis_preds[i], cmap='hot')
        axes[1, i].set_title(f"Sample {i+1}: Prediction (>0.5)")
        axes[1, i].axis('off')

    plt.suptitle("Sample Segmentation Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    print("Saved sample visualization to 'sample_predictions.png'")

    # Figure 2: Aggregate Metrics & IoU Distribution Plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    metrics_names = ['Mean IoU', 'Dice (F1)', 'Precision', 'Recall']
    metrics_values = [m_iou, m_dice, m_prec, m_rec]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    bars = ax1.bar(metrics_names, metrics_values, color=colors, width=0.5)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Overall Model Evaluation Metrics")
    ax1.set_ylabel("Score (0.0 - 1.0)")
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')

    ax2.hist(all_iou, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    ax2.set_title("IoU Score Distribution Across Samples")
    ax2.set_xlabel("IoU Score")
    ax2.set_ylabel("Sample Count")

    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    print("Saved metrics summary plot to 'evaluation_metrics.png'")

if __name__ == "__main__":
    evaluate_model(max_samples=2000)