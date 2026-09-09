import sys
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

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

OUTPUT_FILE = (
    DATA_ROOT
    / "clean_64x256_comparison.png"
)


# ============================================================
# CONFIG
# ============================================================

LR_CROP = 64
SCALE = 4
HR_CROP = LR_CROP * SCALE

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD SPECTRAL MAPPING
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
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print("=" * 70)
print("CLEAN 64×256 REAL SR COMPARISON")
print("=" * 70)

print(
    "\nDevice:",
    DEVICE,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print(
    "\nCheckpoint epoch:",
    checkpoint.get("epoch"),
)

print(
    "Validation loss:",
    checkpoint.get("val_loss"),
)


# ============================================================
# CHOOSE TEST ROI
# ============================================================

test_files = sorted(
    TEST_LR_DIR.glob("*.npy")
)

if not test_files:

    raise RuntimeError(
        "No clean test ROIs found."
    )


# We'll use ROI_0001 if present; otherwise first ROI.
preferred = TEST_LR_DIR / "ROI_0001.npy"

if preferred.exists():

    selected = preferred

else:

    selected = test_files[0]


print(
    "\nSelected ROI:",
    selected.stem,
)


# ============================================================
# LOAD FULL ROI
# ============================================================

lr_full = np.load(
    selected
).astype(np.float32)

hr_full = np.load(
    TEST_HR_DIR / selected.name
).astype(np.float32)


if lr_full.shape != (
    121,
    121,
    4,
):

    raise RuntimeError(
        f"Unexpected LR shape: {lr_full.shape}"
    )


if hr_full.shape != (
    484,
    484,
    4,
):

    raise RuntimeError(
        f"Unexpected HR shape: {hr_full.shape}"
    )


# ============================================================
# APPLY THE SAME CLEAN HARMONIZATION USED IN TRAINING
# ============================================================

lr_harmonized = (
    lr_full
    * slopes.reshape(1, 1, 4)
    + intercepts.reshape(1, 1, 4)
)

lr_harmonized = np.clip(
    lr_harmonized,
    0.0,
    1.0,
).astype(np.float32)


# ============================================================
# SELECT A 64×64 LR CROP
# ============================================================

# Center crop.
y = (
    lr_harmonized.shape[0]
    - LR_CROP
) // 2

x = (
    lr_harmonized.shape[1]
    - LR_CROP
) // 2


lr_crop = lr_harmonized[
    y:y + LR_CROP,
    x:x + LR_CROP,
    :
]


# ============================================================
# EXACT CORRESPONDING 256×256 HR CROP
# ============================================================

hr_y = y * SCALE
hr_x = x * SCALE


hr_crop = hr_full[
    hr_y:hr_y + HR_CROP,
    hr_x:hr_x + HR_CROP,
    :
]


if lr_crop.shape != (
    64,
    64,
    4,
):

    raise RuntimeError(
        f"Bad LR crop: {lr_crop.shape}"
    )


if hr_crop.shape != (
    256,
    256,
    4,
):

    raise RuntimeError(
        f"Bad HR crop: {hr_crop.shape}"
    )


# ============================================================
# CNN
# ============================================================

tensor = torch.from_numpy(
    lr_crop.transpose(2, 0, 1)
).unsqueeze(0).to(DEVICE)


with torch.no_grad():

    sr = model(
        tensor
    )


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


# ============================================================
# METRICS
# ============================================================

cnn_psnr = peak_signal_noise_ratio(
    hr_crop,
    sr,
    data_range=1.0,
)

cnn_ssim = structural_similarity(
    hr_crop,
    sr,
    channel_axis=2,
    data_range=1.0,
)


# ============================================================
# RGB DISPLAY
# ============================================================

def rgb_display(image):

    # B04, B03, B02
    rgb = image[:, :, :3].astype(
        np.float32
    )

    # Use ONE common scale per image.
    # This is display processing only.
    output = np.zeros_like(
        rgb
    )

    for c in range(3):

        channel = rgb[:, :, c]

        low = np.percentile(
            channel,
            2,
        )

        high = np.percentile(
            channel,
            98,
        )

        if high <= low:

            output[:, :, c] = np.clip(
                channel,
                0,
                1,
            )

        else:

            output[:, :, c] = np.clip(
                (
                    channel - low
                )
                / (
                    high - low
                ),
                0,
                1,
            )

    return output


lr_rgb = rgb_display(
    lr_crop
)

sr_rgb = rgb_display(
    sr
)

hr_rgb = rgb_display(
    hr_crop
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6),
)


# ------------------------------------------------------------
# LR
# ------------------------------------------------------------

axes[0].imshow(
    lr_rgb,
    interpolation="nearest",
)

axes[0].set_title(
    "LR — 64 × 64\n"
    "Sentinel-2 • 10m",
    fontsize=17,
    fontweight="bold",
)


# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

axes[1].imshow(
    sr_rgb,
    interpolation="nearest",
)

axes[1].set_title(
    "CNN SR — 256 × 256\n"
    f"Estimated 2.5m\n"
    f"{cnn_psnr:.2f} dB PSNR | "
    f"{cnn_ssim:.3f} SSIM",
    fontsize=17,
    fontweight="bold",
)


# ------------------------------------------------------------
# HR
# ------------------------------------------------------------

axes[2].imshow(
    hr_rgb,
    interpolation="nearest",
)

axes[2].set_title(
    "HR Target — 256 × 256\n"
    "NAIP • 2.5m",
    fontsize=17,
    fontweight="bold",
)


# ------------------------------------------------------------
# Remove axes
# ------------------------------------------------------------

for ax in axes:

    ax.axis("off")


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Real SEN2NAIP Super-Resolution\n"
    "Same Ground Area • 10m → 2.5m",
    fontsize=21,
    fontweight="bold",
)


plt.tight_layout(
    rect=(0, 0, 1, 0.91),
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    OUTPUT_FILE,
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    "\nROI:",
    selected.stem,
)

print(
    "LR crop:",
    lr_crop.shape,
)

print(
    "CNN output:",
    sr.shape,
)

print(
    "HR target:",
    hr_crop.shape,
)

print(
    "\nCNN PSNR:",
    f"{cnn_psnr:.4f} dB",
)

print(
    "CNN SSIM:",
    f"{cnn_ssim:.4f}",
)

print(
    "\nSaved:",
    OUTPUT_FILE,
)

print(
    "\n✅ Same-ground-area comparison complete."
)