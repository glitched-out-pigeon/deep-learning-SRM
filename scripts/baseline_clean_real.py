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
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
# FILES
# ============================================================

lr_files = sorted(
    TEST_LR_DIR.glob("*.npy")
)

if not lr_files:

    raise RuntimeError(
        "No clean test files found."
    )


# ============================================================
# METRICS
# ============================================================

raw_psnr = []
raw_ssim = []

harm_psnr = []
harm_ssim = []


# ============================================================
# EVALUATION
# ============================================================

print("=" * 70)
print("LEAK-FREE SEN2NAIP BICUBIC BASELINE")
print("=" * 70)

print(
    "\nDevice:",
    device,
)

print(
    "Test pairs:",
    len(lr_files),
)


for index, lr_path in enumerate(
    lr_files,
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

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    x = torch.from_numpy(
        lr.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Raw bicubic
    # --------------------------------------------------------

    with torch.no_grad():

        raw = F.interpolate(
            x,
            size=(484, 484),
            mode="bicubic",
            align_corners=False,
        )

    raw = (
        raw
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    raw = np.clip(
        raw,
        0,
        1,
    )

    # --------------------------------------------------------
    # Harmonize
    # --------------------------------------------------------

    mapped = (
        lr
        * slopes.reshape(1, 1, 4)
        + intercepts.reshape(1, 1, 4)
    )

    mapped = np.clip(
        mapped,
        0,
        1,
    ).astype(np.float32)

    mapped_tensor = torch.from_numpy(
        mapped.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Harmonized bicubic
    # --------------------------------------------------------

    with torch.no_grad():

        harmonized = F.interpolate(
            mapped_tensor,
            size=(484, 484),
            mode="bicubic",
            align_corners=False,
        )

    harmonized = (
        harmonized
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    harmonized = np.clip(
        harmonized,
        0,
        1,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    r_psnr = peak_signal_noise_ratio(
        hr,
        raw,
        data_range=1.0,
    )

    r_ssim = structural_similarity(
        hr,
        raw,
        channel_axis=2,
        data_range=1.0,
    )

    h_psnr = peak_signal_noise_ratio(
        hr,
        harmonized,
        data_range=1.0,
    )

    h_ssim = structural_similarity(
        hr,
        harmonized,
        channel_axis=2,
        data_range=1.0,
    )

    raw_psnr.append(r_psnr)
    raw_ssim.append(r_ssim)

    harm_psnr.append(h_psnr)
    harm_ssim.append(h_ssim)

    if (
        index < 5
        or (index + 1) % 50 == 0
    ):

        print(
            f"{index + 1:3d}/{len(lr_files)} "
            f"{lr_path.stem}: "
            f"Raw "
            f"{r_psnr:.4f}/{r_ssim:.4f} | "
            f"Harmonized "
            f"{h_psnr:.4f}/{h_ssim:.4f}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

raw_psnr_mean = np.mean(
    raw_psnr
)

raw_ssim_mean = np.mean(
    raw_ssim
)

harm_psnr_mean = np.mean(
    harm_psnr
)

harm_ssim_mean = np.mean(
    harm_ssim
)


print("\n" + "=" * 70)
print("CLEAN REAL-DATA BASELINE")
print("=" * 70)

print(
    "\nRaw Bicubic ×4:"
)

print(
    f"PSNR: {raw_psnr_mean:.4f} dB"
)

print(
    f"SSIM: {raw_ssim_mean:.4f}"
)

print(
    "\nHarmonized Bicubic ×4:"
)

print(
    f"PSNR: {harm_psnr_mean:.4f} dB"
)

print(
    f"SSIM: {harm_ssim_mean:.4f}"
)

print(
    "\nHarmonization gain:"
)

print(
    f"PSNR: "
    f"{harm_psnr_mean - raw_psnr_mean:+.4f} dB"
)

print(
    f"SSIM: "
    f"{harm_ssim_mean - raw_ssim_mean:+.4f}"
)

print(
    "\n✅ Clean baseline complete."
)