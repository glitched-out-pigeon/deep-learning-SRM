from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "pixelshuffle_cnn_best.pth"
)

TEST_LR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "LR"
)

TEST_HR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "HR"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_visual_comparison.png"
)


# ============================================================
# Add project root to import path
# ============================================================

sys.path.insert(0, str(PROJECT_ROOT))

from models.sr_cnn import SRCNN4x


# ============================================================
# Configuration
# ============================================================

# Use one representative held-out Rajasthan patch.
SAMPLE_NAME = "rajasthan_20260627_0012.npy"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FINAL SUPER-RESOLUTION VISUAL COMPARISON")
print("=" * 70)

print("\nDevice:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Check files
# ============================================================

lr_path = TEST_LR_DIR / SAMPLE_NAME
hr_path = TEST_HR_DIR / SAMPLE_NAME

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

if not lr_path.exists():
    raise FileNotFoundError(
        f"LR file not found:\n{lr_path}"
    )

if not hr_path.exists():
    raise FileNotFoundError(
        f"HR file not found:\n{hr_path}"
    )


# ============================================================
# Load model
# ============================================================

model = SRCNN4x().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "\nCheckpoint epoch:",
    checkpoint["epoch"],
)

print(
    "Validation loss:",
    f"{checkpoint['val_loss']:.6f}",
)


# ============================================================
# Load data
# ============================================================

lr = np.load(lr_path).astype(np.float32)
hr = np.load(hr_path).astype(np.float32)

print("\nInput:")
print("LR shape:", lr.shape)
print("HR shape:", hr.shape)


# ============================================================
# Convert LR to PyTorch tensor
# ============================================================

lr_tensor = torch.from_numpy(
    lr.transpose(2, 0, 1)
).unsqueeze(0)

lr_tensor = lr_tensor.to(device)


# ============================================================
# Generate Bicubic baseline + CNN output
# ============================================================

with torch.no_grad():

    bicubic_tensor = F.interpolate(
        lr_tensor,
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    )

    cnn_tensor = model(lr_tensor)


# ============================================================
# Convert to NumPy HWC
# ============================================================

bicubic = (
    bicubic_tensor
    .squeeze(0)
    .cpu()
    .numpy()
    .transpose(1, 2, 0)
)

cnn = (
    cnn_tensor
    .squeeze(0)
    .cpu()
    .numpy()
    .transpose(1, 2, 0)
)


# Clamp only for evaluation/display.
bicubic = np.clip(bicubic, 0.0, 1.0)
cnn = np.clip(cnn, 0.0, 1.0)
hr = np.clip(hr, 0.0, 1.0)
lr = np.clip(lr, 0.0, 1.0)


# ============================================================
# Calculate metrics for this sample
# ============================================================

bicubic_psnr = peak_signal_noise_ratio(
    hr,
    bicubic,
    data_range=1.0,
)

bicubic_ssim = structural_similarity(
    hr,
    bicubic,
    channel_axis=2,
    data_range=1.0,
)

cnn_psnr = peak_signal_noise_ratio(
    hr,
    cnn,
    data_range=1.0,
)

cnn_ssim = structural_similarity(
    hr,
    cnn,
    channel_axis=2,
    data_range=1.0,
)


print("\nSample metrics:")

print(
    f"Bicubic: PSNR={bicubic_psnr:.4f} dB | "
    f"SSIM={bicubic_ssim:.4f}"
)

print(
    f"CNN:     PSNR={cnn_psnr:.4f} dB | "
    f"SSIM={cnn_ssim:.4f}"
)


# ============================================================
# Create clean presentation figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10),
)

# ------------------------------------------------------------
# LR
# ------------------------------------------------------------

axes[0, 0].imshow(
    lr,
    interpolation="nearest",
)

axes[0, 0].set_title(
    "Input LR\n64 × 64"
)

axes[0, 0].axis("off")


# ------------------------------------------------------------
# Bicubic
# ------------------------------------------------------------

axes[0, 1].imshow(
    bicubic,
    interpolation="nearest",
)

axes[0, 1].set_title(
    f"Bicubic ×4\n"
    f"PSNR: {bicubic_psnr:.2f} dB | "
    f"SSIM: {bicubic_ssim:.4f}"
)

axes[0, 1].axis("off")


# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

axes[1, 0].imshow(
    cnn,
    interpolation="nearest",
)

axes[1, 0].set_title(
    f"Residual PixelShuffle CNN\n"
    f"PSNR: {cnn_psnr:.2f} dB | "
    f"SSIM: {cnn_ssim:.4f}"
)

axes[1, 0].axis("off")


# ------------------------------------------------------------
# HR
# ------------------------------------------------------------

axes[1, 1].imshow(
    hr,
    interpolation="nearest",
)

axes[1, 1].set_title(
    "Reference HR\n256 × 256"
)

axes[1, 1].axis("off")


# ------------------------------------------------------------
# Overall title
# ------------------------------------------------------------

fig.suptitle(
    "Sentinel-2 4× Super-Resolution — Held-Out Rajasthan Scene",
    fontsize=16,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved presentation figure:")
print(OUTPUT_PATH)

print("\n✅ Final visualization complete!")