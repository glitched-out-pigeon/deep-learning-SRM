import sys
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.sr_cnn import ResidualPixelShuffleSR


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_clean"
)

TEST_LR_DIR = DATA_ROOT / "test" / "LR"
TEST_HR_DIR = DATA_ROOT / "test" / "HR"

MAPPING_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "real_clean"
    / "clean_spectral_mapping.json"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "real_clean"
    / "sen2naip_clean_harmonized_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MAPPING
# ============================================================

with open(
    MAPPING_FILE,
    "r",
    encoding="utf-8",
) as f:

    mapping = json.load(f)

slopes = np.asarray(
    mapping["slopes"],
    dtype=np.float32,
)

intercepts = np.asarray(
    mapping["intercepts"],
    dtype=np.float32,
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


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# FILES
# ============================================================

lr_files = sorted(
    TEST_LR_DIR.glob("*.npy")
)

if not lr_files:
    raise RuntimeError(
        "No clean test LR files found."
    )

print("=" * 70)
print("LEAK-FREE SEN2NAIP CNN EVALUATION")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "\nCheckpoint epoch:",
    checkpoint.get("epoch")
)

print(
    "Validation loss:",
    checkpoint.get("val_loss")
)

print(
    "\nTest pairs:",
    len(lr_files)
)


# ============================================================
# STORAGE
# ============================================================

bicubic_psnr = []
bicubic_ssim = []

cnn_psnr = []
cnn_ssim = []


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for index, lr_path in enumerate(
        lr_files
    ):

        hr_path = (
            TEST_HR_DIR
            / lr_path.name
        )

        lr = np.load(
            lr_path
        ).astype(np.float32)

        hr = np.load(
            hr_path
        ).astype(np.float32)

        # ----------------------------------------------------
        # Apply clean train-only spectral mapping
        # ----------------------------------------------------

        harmonized = (
            lr
            * slopes.reshape(1, 1, 4)
            + intercepts.reshape(1, 1, 4)
        )

        harmonized = np.clip(
            harmonized,
            0.0,
            1.0,
        ).astype(np.float32)

        x = torch.from_numpy(
            harmonized.transpose(2, 0, 1)
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Harmonized bicubic
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            size=(484, 484),
            mode="bicubic",
            align_corners=False,
        )

        bicubic = (
            bicubic
            .squeeze(0)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        bicubic = np.clip(
            bicubic,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        sr = model(x)

        sr = (
            sr
            .squeeze(0)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        sr = np.clip(
            sr,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        b_psnr = peak_signal_noise_ratio(
            hr,
            bicubic,
            data_range=1.0,
        )

        b_ssim = structural_similarity(
            hr,
            bicubic,
            channel_axis=2,
            data_range=1.0,
        )

        c_psnr = peak_signal_noise_ratio(
            hr,
            sr,
            data_range=1.0,
        )

        c_ssim = structural_similarity(
            hr,
            sr,
            channel_axis=2,
            data_range=1.0,
        )

        bicubic_psnr.append(b_psnr)
        bicubic_ssim.append(b_ssim)

        cnn_psnr.append(c_psnr)
        cnn_ssim.append(c_ssim)

        if (
            index < 10
            or (index + 1) % 50 == 0
        ):

            print(
                f"{index + 1:3d}/{len(lr_files)} "
                f"{lr_path.stem}: "
                f"Bicubic "
                f"{b_psnr:.4f}/{b_ssim:.4f} | "
                f"CNN "
                f"{c_psnr:.4f}/{c_ssim:.4f}"
            )


# ============================================================
# FINAL STATISTICS
# ============================================================

b_psnr_mean = np.mean(bicubic_psnr)
b_psnr_std = np.std(bicubic_psnr)

b_ssim_mean = np.mean(bicubic_ssim)
b_ssim_std = np.std(bicubic_ssim)

c_psnr_mean = np.mean(cnn_psnr)
c_psnr_std = np.std(cnn_psnr)

c_ssim_mean = np.mean(cnn_ssim)
c_ssim_std = np.std(cnn_ssim)


psnr_gain = (
    c_psnr_mean
    - b_psnr_mean
)

ssim_gain = (
    c_ssim_mean
    - b_ssim_mean
)


psnr_gains = (
    np.asarray(cnn_psnr)
    - np.asarray(bicubic_psnr)
)

ssim_gains = (
    np.asarray(cnn_ssim)
    - np.asarray(bicubic_ssim)
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL LEAK-FREE REAL-DATA RESULTS")
print("=" * 70)

print("\nHARMONIZED BICUBIC ×4")

print(
    f"PSNR: "
    f"{b_psnr_mean:.4f} ± "
    f"{b_psnr_std:.4f} dB"
)

print(
    f"SSIM: "
    f"{b_ssim_mean:.4f} ± "
    f"{b_ssim_std:.4f}"
)

print("\nCLEAN HARMONIZED CNN ×4")

print(
    f"PSNR: "
    f"{c_psnr_mean:.4f} ± "
    f"{c_psnr_std:.4f} dB"
)

print(
    f"SSIM: "
    f"{c_ssim_mean:.4f} ± "
    f"{c_ssim_std:.4f}"
)

print("\nCNN IMPROVEMENT")

print(
    f"PSNR: "
    f"{psnr_gain:+.4f} dB"
)

print(
    f"SSIM: "
    f"{ssim_gain:+.4f}"
)

print("\nTEST-ROI DISTRIBUTION")

print(
    "CNN improves PSNR:",
    f"{np.sum(psnr_gains > 0)}/{len(psnr_gains)} "
    f"({100 * np.mean(psnr_gains > 0):.1f}%)"
)

print(
    "CNN improves SSIM:",
    f"{np.sum(ssim_gains > 0)}/{len(ssim_gains)} "
    f"({100 * np.mean(ssim_gains > 0):.1f}%)"
)

print(
    "\nMedian PSNR gain:",
    f"{np.median(psnr_gains):+.4f} dB"
)

print(
    "Min PSNR gain:",
    f"{np.min(psnr_gains):+.4f} dB"
)

print(
    "Max PSNR gain:",
    f"{np.max(psnr_gains):+.4f} dB"
)

print("\n" + "-" * 70)

if (
    psnr_gain > 0
    and ssim_gain > 0
):

    print(
        "✅ CNN improves BOTH PSNR and SSIM."
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
        "❌ CNN does not beat bicubic."
    )

print("-" * 70)

print(
    "\n✅ Leak-free evaluation complete."
)