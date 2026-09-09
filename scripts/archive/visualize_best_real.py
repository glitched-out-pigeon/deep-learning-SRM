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
    / "real"
)

LR_DIR = DATA_ROOT / "test" / "LR"
HR_DIR = DATA_ROOT / "test" / "HR"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "real"
    / "sen2naip_harmonized_best.pth"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "real"
    / "sen2naip_spectral_mapping.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
    / "best_real_srm_comparison.png"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# DISPLAY NORMALIZATION
# ============================================================

def rgb_stretch(image):
    """
    Convert RGB data to a visually useful [0,1] representation.

    Uses a 2-98 percentile stretch independently per channel.
    """

    rgb = image[:, :, :3].astype(np.float32)

    output = np.zeros_like(rgb)

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
                (channel - low)
                / (high - low),
                0,
                1,
            )

    return output


# ============================================================
# RUN ONE ROI
# ============================================================

def process_roi(lr_path):

    hr_path = HR_DIR / lr_path.name

    lr = np.load(
        lr_path
    ).astype(np.float32)

    hr = np.load(
        hr_path
    ).astype(np.float32)

    # --------------------------------------------------------
    # Raw bicubic
    # --------------------------------------------------------

    raw_tensor = torch.from_numpy(
        lr.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        raw_bicubic = F.interpolate(
            raw_tensor,
            size=(484, 484),
            mode="bicubic",
            align_corners=False,
        )

    raw_bicubic = (
        raw_bicubic
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    raw_bicubic = np.clip(
        raw_bicubic,
        0,
        1,
    )

    # --------------------------------------------------------
    # Spectral harmonization
    # --------------------------------------------------------

    harmonized = (
        lr
        * slopes.reshape(1, 1, 4)
        + intercepts.reshape(1, 1, 4)
    )

    harmonized = np.clip(
        harmonized,
        0,
        1,
    ).astype(np.float32)

    harmonized_tensor = torch.from_numpy(
        harmonized.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Harmonized bicubic
    # --------------------------------------------------------

    with torch.no_grad():

        harmonized_bicubic = F.interpolate(
            harmonized_tensor,
            size=(484, 484),
            mode="bicubic",
            align_corners=False,
        )

    harmonized_bicubic = (
        harmonized_bicubic
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    harmonized_bicubic = np.clip(
        harmonized_bicubic,
        0,
        1,
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            harmonized_tensor
        )

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    prediction = np.clip(
        prediction,
        0,
        1,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    bicubic_psnr = peak_signal_noise_ratio(
        hr,
        harmonized_bicubic,
        data_range=1.0,
    )

    bicubic_ssim = structural_similarity(
        hr,
        harmonized_bicubic,
        channel_axis=2,
        data_range=1.0,
    )

    cnn_psnr = peak_signal_noise_ratio(
        hr,
        prediction,
        data_range=1.0,
    )

    cnn_ssim = structural_similarity(
        hr,
        prediction,
        channel_axis=2,
        data_range=1.0,
    )

    return {
        "name": lr_path.stem,
        "lr": lr,
        "raw_bicubic": raw_bicubic,
        "harmonized": harmonized,
        "harmonized_bicubic": harmonized_bicubic,
        "cnn": prediction,
        "hr": hr,
        "bicubic_psnr": bicubic_psnr,
        "bicubic_ssim": bicubic_ssim,
        "cnn_psnr": cnn_psnr,
        "cnn_ssim": cnn_ssim,
        "psnr_gain": cnn_psnr - bicubic_psnr,
        "ssim_gain": cnn_ssim - bicubic_ssim,
    }


# ============================================================
# FIND BEST / MEDIAN / WORST
# ============================================================

print("=" * 70)
print("REAL SEN2NAIP VISUAL TEST")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

lr_files = sorted(
    LR_DIR.glob("*.npy")
)

print(
    "\nTesting",
    len(lr_files),
    "held-out ROIs..."
)

results = []

with torch.no_grad():

    for index, lr_path in enumerate(lr_files):

        result = process_roi(
            lr_path
        )

        results.append(result)

        if (
            index < 5
            or (index + 1) % 50 == 0
        ):
            print(
                f"{index + 1:3d}/{len(lr_files)} "
                f"{lr_path.stem}: "
                f"gain="
                f"{result['psnr_gain']:+.3f} dB"
            )


# ============================================================
# SORT BY PSNR GAIN
# ============================================================

results.sort(
    key=lambda x: x["psnr_gain"]
)

worst = results[0]

median = results[
    len(results) // 2
]

best = results[-1]


# ============================================================
# PRINT SELECTED ROIS
# ============================================================

print("\n" + "=" * 70)
print("SELECTED VISUAL CASES")
print("=" * 70)

for label, result in [
    ("WORST", worst),
    ("MEDIAN", median),
    ("BEST", best),
]:

    print(
        f"\n{label}: {result['name']}"
    )

    print(
        f"  Bicubic: "
        f"{result['bicubic_psnr']:.4f} dB / "
        f"{result['bicubic_ssim']:.4f}"
    )

    print(
        f"  CNN:     "
        f"{result['cnn_psnr']:.4f} dB / "
        f"{result['cnn_ssim']:.4f}"
    )

    print(
        f"  Gain:    "
        f"{result['psnr_gain']:+.4f} dB / "
        f"{result['ssim_gain']:+.4f}"
    )


# ============================================================
# CREATE FIGURE
# ============================================================

cases = [
    ("BEST CASE", best),
    ("TYPICAL CASE", median),
    ("WORST CASE", worst),
]

fig, axes = plt.subplots(
    3,
    5,
    figsize=(20, 12),
)

for row, (label, result) in enumerate(cases):

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    raw_input = rgb_stretch(
        result["raw_bicubic"]
    )

    harmonized_bicubic = rgb_stretch(
        result["harmonized_bicubic"]
    )

    cnn = rgb_stretch(
        result["cnn"]
    )

    hr = rgb_stretch(
        result["hr"]
    )

    lr_rgb = rgb_stretch(
        result["harmonized"]
    )

    # --------------------------------------------------------
    # Panel 1
    # --------------------------------------------------------

    axes[row, 0].imshow(
        lr_rgb
    )

    axes[row, 0].set_title(
        "10m S2\n(Harmonized)"
    )

    # --------------------------------------------------------
    # Panel 2
    # --------------------------------------------------------

    axes[row, 1].imshow(
        raw_input
    )

    axes[row, 1].set_title(
        "Bicubic ×4\nRaw S2"
    )

    # --------------------------------------------------------
    # Panel 3
    # --------------------------------------------------------

    axes[row, 2].imshow(
        harmonized_bicubic
    )

    axes[row, 2].set_title(
        f"Harmonized Bicubic\n"
        f"{result['bicubic_psnr']:.2f} dB / "
        f"{result['bicubic_ssim']:.3f}"
    )

    # --------------------------------------------------------
    # Panel 4
    # --------------------------------------------------------

    axes[row, 3].imshow(
        cnn
    )

    axes[row, 3].set_title(
        f"OUR CNN\n"
        f"{result['cnn_psnr']:.2f} dB / "
        f"{result['cnn_ssim']:.3f}"
    )

    # --------------------------------------------------------
    # Panel 5
    # --------------------------------------------------------

    axes[row, 4].imshow(
        hr
    )

    axes[row, 4].set_title(
        "NAIP 2.5m\nREFERENCE"
    )

    # --------------------------------------------------------
    # Remove axes
    # --------------------------------------------------------

    for col in range(5):

        axes[row, col].axis(
            "off"
        )

    # --------------------------------------------------------
    # Row label
    # --------------------------------------------------------

    axes[row, 0].set_ylabel(
        label,
        fontsize=12,
        fontweight="bold",
    )


# ============================================================
# FIGURE TITLE
# ============================================================

fig.suptitle(
    "SEN2NAIP Real-Data Super-Resolution Test\n"
    "10m Sentinel-2 → 2.5m Estimated Output",
    fontsize=18,
    fontweight="bold",
)

plt.tight_layout(
    rect=(0, 0, 1, 0.95)
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
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(
    "\nSaved:",
    OUTPUT_FILE
)

print(
    "\nBEST CASE:",
    best["name"]
)

print(
    f"PSNR gain: "
    f"{best['psnr_gain']:+.4f} dB"
)

print(
    f"SSIM gain: "
    f"{best['ssim_gain']:+.4f}"
)

print("\n✅ Comparison image created.")