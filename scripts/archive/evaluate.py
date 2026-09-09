from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Add project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.sr_cnn import SRCNN4x
from dataset import SuperResolutionDataset


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "pixelshuffle_cnn_best.pth"

BATCH_SIZE = 1


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SUPER-RESOLUTION EVALUATION")
print("=" * 70)

print("\nDevice:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Dataset
# ============================================================

test_dataset = SuperResolutionDataset(
    root_dir=DATA_ROOT,
    split="test",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Load CNN
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
    "\nLoaded checkpoint from epoch:",
    checkpoint["epoch"],
)

print(
    "Checkpoint validation loss:",
    f"{checkpoint['val_loss']:.6f}",
)


# ============================================================
# Evaluation
# ============================================================

bicubic_psnr_values = []
bicubic_ssim_values = []

cnn_psnr_values = []
cnn_ssim_values = []


print("\n" + "-" * 70)
print("PER-IMAGE RESULTS")
print("-" * 70)

with torch.no_grad():

    for index, (lr, hr) in enumerate(test_loader):

        lr = lr.to(device)
        hr = hr.to(device)

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(
            lr,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
        )

        bicubic = torch.clamp(
            bicubic,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # CNN prediction
        # ----------------------------------------------------

        sr = model(lr)

        sr = torch.clamp(
            sr,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # Convert to HWC NumPy images
        # ----------------------------------------------------

        hr_img = (
            hr[0]
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        bicubic_img = (
            bicubic[0]
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        sr_img = (
            sr[0]
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        bicubic_psnr = peak_signal_noise_ratio(
            hr_img,
            bicubic_img,
            data_range=1.0,
        )

        cnn_psnr = peak_signal_noise_ratio(
            hr_img,
            sr_img,
            data_range=1.0,
        )

        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        bicubic_ssim = structural_similarity(
            hr_img,
            bicubic_img,
            channel_axis=2,
            data_range=1.0,
        )

        cnn_ssim = structural_similarity(
            hr_img,
            sr_img,
            channel_axis=2,
            data_range=1.0,
        )

        bicubic_psnr_values.append(bicubic_psnr)
        bicubic_ssim_values.append(bicubic_ssim)

        cnn_psnr_values.append(cnn_psnr)
        cnn_ssim_values.append(cnn_ssim)

        filename = test_dataset.lr_files[index].name

        print(
            f"\n{filename}"
        )

        print(
            f"  Bicubic -> "
            f"PSNR: {bicubic_psnr:.4f} dB | "
            f"SSIM: {bicubic_ssim:.4f}"
        )

        print(
            f"  CNN     -> "
            f"PSNR: {cnn_psnr:.4f} dB | "
            f"SSIM: {cnn_ssim:.4f}"
        )


# ============================================================
# Average metrics
# ============================================================

avg_bicubic_psnr = np.mean(
    bicubic_psnr_values
)

avg_bicubic_ssim = np.mean(
    bicubic_ssim_values
)

avg_cnn_psnr = np.mean(
    cnn_psnr_values
)

avg_cnn_ssim = np.mean(
    cnn_ssim_values
)


# ============================================================
# Final results
# ============================================================

print("\n" + "=" * 70)
print("AVERAGE TEST RESULTS")
print("=" * 70)

print(
    f"\nBicubic:"
    f"\n  PSNR: {avg_bicubic_psnr:.4f} dB"
    f"\n  SSIM: {avg_bicubic_ssim:.4f}"
)

print(
    f"\nCNN:"
    f"\n  PSNR: {avg_cnn_psnr:.4f} dB"
    f"\n  SSIM: {avg_cnn_ssim:.4f}"
)


# ============================================================
# Improvement
# ============================================================

psnr_gain = (
    avg_cnn_psnr - avg_bicubic_psnr
)

ssim_gain = (
    avg_cnn_ssim - avg_bicubic_ssim
)

print("\n" + "-" * 70)

print(
    f"PSNR improvement: "
    f"{psnr_gain:+.4f} dB"
)

print(
    f"SSIM improvement: "
    f"{ssim_gain:+.4f}"
)

print("-" * 70)

if psnr_gain > 0 and ssim_gain > 0:
    print(
        "\n✅ CNN beats bicubic on both metrics."
    )
elif psnr_gain > 0:
    print(
        "\n⚠️ CNN improves PSNR but not SSIM."
    )
elif ssim_gain > 0:
    print(
        "\n⚠️ CNN improves SSIM but not PSNR."
    )
else:
    print(
        "\n❌ CNN does not beat bicubic yet."
    )

print("\nEvaluation complete.")