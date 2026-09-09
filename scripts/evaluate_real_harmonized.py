import sys
from pathlib import Path
import json

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

TEST_LR_DIR = DATA_ROOT / "test" / "LR"
TEST_HR_DIR = DATA_ROOT / "test" / "HR"

MAPPING_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "real"
    / "sen2naip_spectral_mapping.json"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "real"
    / "sen2naip_harmonized_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MAPPING
# ============================================================

if not MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"Mapping not found:\n{MAPPING_FILE}"
    )

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

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

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
        "No test LR files found."
    )


# ============================================================
# METRIC STORAGE
# ============================================================

bicubic_psnr = []
bicubic_ssim = []

cnn_psnr = []
cnn_ssim = []


# ============================================================
# EVALUATION
# ============================================================

print("=" * 70)
print("SEN2NAIP HARMONIZED CNN EVALUATION")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "\nCheckpoint epoch:",
    checkpoint.get("epoch", "unknown")
)

print(
    "Validation loss:",
    checkpoint.get("val_loss", "unknown")
)

print(
    "\nTest pairs:",
    len(lr_files)
)


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
        # Apply TRAIN-ONLY spectral mapping
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

        # HWC -> NCHW
        x = torch.from_numpy(
            harmonized.transpose(2, 0, 1)
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Harmonized bicubic
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
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
            .transpose(1, 2, 0)
        )

        bicubic = np.clip(
            bicubic,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # Harmonized CNN
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
                f"{b_psnr:.4f} / {b_ssim:.4f} | "
                f"CNN "
                f"{c_psnr:.4f} / {c_ssim:.4f}"
            )


# ============================================================
# FINAL NUMBERS
# ============================================================

mean_b_psnr = np.mean(bicubic_psnr)
mean_b_ssim = np.mean(bicubic_ssim)

mean_c_psnr = np.mean(cnn_psnr)
mean_c_ssim = np.mean(cnn_ssim)

psnr_gain = mean_c_psnr - mean_b_psnr
ssim_gain = mean_c_ssim - mean_b_ssim


print("\n" + "=" * 70)
print("FINAL HARMONIZED REAL-DATA RESULTS")
print("=" * 70)

print("\nHarmonized Bicubic ×4")
print(
    f"PSNR: {mean_b_psnr:.4f} dB"
)
print(
    f"SSIM: {mean_b_ssim:.4f}"
)

print("\nHarmonized CNN ×4")
print(
    f"PSNR: {mean_c_psnr:.4f} dB"
)
print(
    f"SSIM: {mean_c_ssim:.4f}"
)

print("\nCNN GAIN OVER HARMONIZED BICUBIC")
print(
    f"PSNR: {psnr_gain:+.4f} dB"
)
print(
    f"SSIM: {ssim_gain:+.4f}"
)

print("\n" + "-" * 70)

if psnr_gain > 0 and ssim_gain > 0:
    print(
        "✅ CNN beats harmonized bicubic on BOTH metrics."
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
        "❌ CNN does not beat harmonized bicubic."
    )

print("-" * 70)
print("\n✅ Evaluation complete.")