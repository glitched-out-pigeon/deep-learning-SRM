import sys
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
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

LR_DIR = DATA_ROOT / "test" / "LR"
HR_DIR = DATA_ROOT / "test" / "HR"

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
    / "final_clean_srm_comparison.png"
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
# LOAD SPECTRAL MAPPING
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
# RGB DISPLAY
# ============================================================

def make_rgb(image):
    """
    Use B04/B03/B02 for RGB display.
    """

    rgb = image[:, :, :3].astype(
        np.float32
    )

    # Percentile stretch
    output = np.zeros_like(rgb)

    for c in range(3):

        channel = rgb[:, :, c]

        low = np.percentile(
            channel,
            2
        )

        high = np.percentile(
            channel,
            98
        )

        if high <= low:

            output[:, :, c] = np.clip(
                channel,
                0,
                1
            )

        else:

            output[:, :, c] = np.clip(
                (channel - low)
                / (high - low),
                0,
                1
            )

    return output


# ============================================================
# FIND REPRESENTATIVE ROI
# ============================================================

test_files = sorted(
    LR_DIR.glob("*.npy")
)

if not test_files:
    raise RuntimeError(
        "No clean test files found."
    )


# We select the ROI whose CNN-vs-bicubic
# gain is closest to the median.
candidate_results = []


print("=" * 70)
print("SELECTING REPRESENTATIVE CLEAN TEST ROI")
print("=" * 70)

for lr_path in test_files:

    hr_path = HR_DIR / lr_path.name

    lr = np.load(
        lr_path
    ).astype(np.float32)

    hr = np.load(
        hr_path
    ).astype(np.float32)

    # Harmonize
    harmonized = (
        lr
        * slopes.reshape(1, 1, 4)
        + intercepts.reshape(1, 1, 4)
    )

    harmonized = np.clip(
        harmonized,
        0,
        1
    )

    x = torch.from_numpy(
        harmonized.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        bicubic = F.interpolate(
            x,
            size=(484, 484),
            mode="bicubic",
            align_corners=False
        )

        sr = model(x)

    bicubic = (
        bicubic
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    sr = (
        sr
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    bicubic = np.clip(
        bicubic,
        0,
        1
    )

    sr = np.clip(
        sr,
        0,
        1
    )

    bicubic_psnr = peak_signal_noise_ratio(
        hr,
        bicubic,
        data_range=1.0
    )

    cnn_psnr = peak_signal_noise_ratio(
        hr,
        sr,
        data_range=1.0
    )

    gain = (
        cnn_psnr
        - bicubic_psnr
    )

    candidate_results.append(
        (
            gain,
            lr_path
        )
    )


# Median gain ROI
candidate_results.sort(
    key=lambda x: x[0]
)

selected_gain, selected_path = (
    candidate_results[
        len(candidate_results) // 2
    ]
)


print(
    "\nSelected ROI:",
    selected_path.stem
)

print(
    "Approximate PSNR gain:",
    f"{selected_gain:+.4f} dB"
)


# ============================================================
# LOAD SELECTED ROI
# ============================================================

lr = np.load(
    selected_path
).astype(np.float32)

hr = np.load(
    HR_DIR / selected_path.name
).astype(np.float32)


# ============================================================
# HARMONIZE INPUT
# ============================================================

harmonized = (
    lr
    * slopes.reshape(1, 1, 4)
    + intercepts.reshape(1, 1, 4)
)

harmonized = np.clip(
    harmonized,
    0,
    1
).astype(np.float32)


x = torch.from_numpy(
    harmonized.transpose(2, 0, 1)
).unsqueeze(0).to(device)


# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():

    bicubic = F.interpolate(
        x,
        size=(484, 484),
        mode="bicubic",
        align_corners=False
    )

    sr = model(x)


bicubic = (
    bicubic
    .squeeze(0)
    .cpu()
    .numpy()
    .transpose(1, 2, 0)
)

sr = (
    sr
    .squeeze(0)
    .cpu()
    .numpy()
    .transpose(1, 2, 0)
)

bicubic = np.clip(
    bicubic,
    0,
    1
)

sr = np.clip(
    sr,
    0,
    1
)


# ============================================================
# METRICS
# ============================================================

bicubic_psnr = peak_signal_noise_ratio(
    hr,
    bicubic,
    data_range=1.0
)

bicubic_ssim = structural_similarity(
    hr,
    bicubic,
    channel_axis=2,
    data_range=1.0
)

cnn_psnr = peak_signal_noise_ratio(
    hr,
    sr,
    data_range=1.0
)

cnn_ssim = structural_similarity(
    hr,
    sr,
    channel_axis=2,
    data_range=1.0
)


# ============================================================
# ZOOM
# ============================================================

zoom_size = 256

center_y = 484 // 2
center_x = 484 // 2

half = zoom_size // 2

y1 = center_y - half
y2 = center_y + half

x1 = center_x - half
x2 = center_x + half


bicubic_zoom = bicubic[
    y1:y2,
    x1:x2
]

cnn_zoom = sr[
    y1:y2,
    x1:x2
]

hr_zoom = hr[
    y1:y2,
    x1:x2
]


# ============================================================
# DISPLAY IMAGES
# ============================================================

lr_rgb = make_rgb(
    harmonized
)

bicubic_rgb = make_rgb(
    bicubic
)

cnn_rgb = make_rgb(
    sr
)

hr_rgb = make_rgb(
    hr
)

bicubic_zoom_rgb = make_rgb(
    bicubic_zoom
)

cnn_zoom_rgb = make_rgb(
    cnn_zoom
)

hr_zoom_rgb = make_rgb(
    hr_zoom
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10)
)


# ------------------------------------------------------------
# TOP ROW
# ------------------------------------------------------------

axes[0, 0].imshow(
    lr_rgb
)

axes[0, 0].set_title(
    "10m Sentinel-2 Input\n121 × 121"
)


axes[0, 1].imshow(
    cnn_rgb
)

axes[0, 1].set_title(
    f"Our CNN SR — 2.5m\n"
    f"{cnn_psnr:.2f} dB PSNR | "
    f"{cnn_ssim:.3f} SSIM"
)


axes[0, 2].imshow(
    hr_rgb
)

axes[0, 2].set_title(
    "NAIP Reference — 2.5m\n"
    "484 × 484"
)


# ------------------------------------------------------------
# BOTTOM ROW
# ------------------------------------------------------------

axes[1, 0].imshow(
    bicubic_zoom_rgb
)

axes[1, 0].set_title(
    f"Bicubic ×4 — Zoom\n"
    f"{bicubic_psnr:.2f} dB | "
    f"{bicubic_ssim:.3f}"
)


axes[1, 1].imshow(
    cnn_zoom_rgb
)

axes[1, 1].set_title(
    "Our CNN — Zoom"
)


axes[1, 2].imshow(
    hr_zoom_rgb
)

axes[1, 2].set_title(
    "NAIP Reference — Zoom"
)


# ------------------------------------------------------------
# Remove axes
# ------------------------------------------------------------

for row in range(2):

    for col in range(3):

        axes[row, col].axis(
            "off"
        )


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Real Satellite Super-Resolution\n"
    "10m Sentinel-2 → 2.5m Estimated Output",
    fontsize=18,
    fontweight="bold"
)


plt.tight_layout(
    rect=(0, 0, 1, 0.94)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEAN VISUAL COMPARISON")
print("=" * 70)

print(
    "\nROI:",
    selected_path.stem
)

print(
    "\nHarmonized Bicubic:"
)

print(
    f"PSNR: {bicubic_psnr:.4f} dB"
)

print(
    f"SSIM: {bicubic_ssim:.4f}"
)

print(
    "\nCNN:"
)

print(
    f"PSNR: {cnn_psnr:.4f} dB"
)

print(
    f"SSIM: {cnn_ssim:.4f}"
)

print(
    "\nImprovement:"
)

print(
    f"PSNR: "
    f"{cnn_psnr - bicubic_psnr:+.4f} dB"
)

print(
    f"SSIM: "
    f"{cnn_ssim - bicubic_ssim:+.4f}"
)

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    "\n✅ Final comparison created."
)