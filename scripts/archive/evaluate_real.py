import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MODEL
# ============================================================

from models.sr_cnn import ResidualPixelShuffleSR


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
)

TEST_LR_DIR = (
    DATA_ROOT
    / "test"
    / "LR"
)

TEST_HR_DIR = (
    DATA_ROOT
    / "test"
    / "HR"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "real"
    / "sen2naip_pixelshuffle_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = ResidualPixelShuffleSR(
    in_channels=4,
    out_channels=4,
    features=64,
    num_blocks=8,
).to(device)


if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# PRINT INFORMATION
# ============================================================

print("=" * 70)
print("REAL SEN2NAIP CNN EVALUATION")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("\nCheckpoint:")
print(CHECKPOINT)

print(
    "\nCheckpoint epoch:",
    checkpoint.get("epoch", "unknown")
)

print(
    "Validation loss:",
    checkpoint.get("val_loss", "unknown")
)


# ============================================================
# TEST FILES
# ============================================================

lr_files = sorted(
    TEST_LR_DIR.glob("*.npy")
)

hr_files = sorted(
    TEST_HR_DIR.glob("*.npy")
)

if len(lr_files) != len(hr_files):
    raise RuntimeError(
        f"LR/HR count mismatch: "
        f"{len(lr_files)} vs {len(hr_files)}"
    )

print(
    "\nTest pairs:",
    len(lr_files)
)


# ============================================================
# METRICS
# ============================================================

cnn_psnr_values = []
cnn_ssim_values = []

bicubic_psnr_values = []
bicubic_ssim_values = []


# ============================================================
# EVALUATE
# ============================================================

with torch.no_grad():

    for index, lr_path in enumerate(lr_files):

        hr_path = TEST_HR_DIR / lr_path.name

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        lr = np.load(
            lr_path
        ).astype(np.float32)

        hr = np.load(
            hr_path
        ).astype(np.float32)

        # ----------------------------------------------------
        # HWC -> NCHW
        # ----------------------------------------------------

        lr_tensor = torch.from_numpy(
            lr.transpose(2, 0, 1)
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # CNN prediction
        # ----------------------------------------------------

        sr = model(lr_tensor)

        sr = sr.squeeze(0).cpu().numpy()

        sr = np.transpose(
            sr,
            (1, 2, 0)
        )

        sr = np.clip(
            sr,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(
            lr_tensor,
            size=(
                hr.shape[0],
                hr.shape[1],
            ),
            mode="bicubic",
            align_corners=False,
        )

        bicubic = (
            bicubic
            .squeeze(0)
            .cpu()
            .numpy()
        )

        bicubic = np.transpose(
            bicubic,
            (1, 2, 0)
        )

        bicubic = np.clip(
            bicubic,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # CNN metrics
        # ----------------------------------------------------

        cnn_psnr = peak_signal_noise_ratio(
            hr,
            sr,
            data_range=1.0,
        )

        cnn_ssim = structural_similarity(
            hr,
            sr,
            channel_axis=2,
            data_range=1.0,
        )

        # ----------------------------------------------------
        # Bicubic metrics
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        cnn_psnr_values.append(
            cnn_psnr
        )

        cnn_ssim_values.append(
            cnn_ssim
        )

        bicubic_psnr_values.append(
            bicubic_psnr
        )

        bicubic_ssim_values.append(
            bicubic_ssim
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index < 10
            or (index + 1) % 50 == 0
        ):

            print(
                f"{index + 1:3d}/{len(lr_files)} "
                f"{lr_path.stem}: "
                f"Bicubic "
                f"{bicubic_psnr:.4f} dB / "
                f"{bicubic_ssim:.4f} | "
                f"CNN "
                f"{cnn_psnr:.4f} dB / "
                f"{cnn_ssim:.4f}"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

mean_bicubic_psnr = np.mean(
    bicubic_psnr_values
)

mean_bicubic_ssim = np.mean(
    bicubic_ssim_values
)

mean_cnn_psnr = np.mean(
    cnn_psnr_values
)

mean_cnn_ssim = np.mean(
    cnn_ssim_values
)


psnr_gain = (
    mean_cnn_psnr
    - mean_bicubic_psnr
)

ssim_gain = (
    mean_cnn_ssim
    - mean_bicubic_ssim
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL REAL-DATA RESULTS")
print("=" * 70)

print("\nBICUBIC ×4")
print(
    f"PSNR: {mean_bicubic_psnr:.4f} dB"
)

print(
    f"SSIM: {mean_bicubic_ssim:.4f}"
)

print("\nCNN ×4")
print(
    f"PSNR: {mean_cnn_psnr:.4f} dB"
)

print(
    f"SSIM: {mean_cnn_ssim:.4f}"
)

print("\nCNN GAIN OVER BICUBIC")

print(
    f"PSNR: {psnr_gain:+.4f} dB"
)

print(
    f"SSIM: {ssim_gain:+.4f}"
)

print("\n" + "-" * 70)

if psnr_gain > 0 and ssim_gain > 0:

    print(
        "✅ CNN beats bicubic on BOTH PSNR and SSIM."
    )

elif psnr_gain > 0:

    print(
        "⚠️ CNN improves PSNR but not SSIM."
    )

elif ssim_gain > 0:

    print(
        "⚠️ CNN improves SSIM but not PSNR."
    )

else:

    print(
        "❌ CNN does not beat bicubic on the current test."
    )

print("-" * 70)

print(
    "\n✅ Real SEN2NAIP evaluation complete."
)