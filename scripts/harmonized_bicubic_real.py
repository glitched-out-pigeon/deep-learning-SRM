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


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
)

TRAIN_LR_DIR = DATA_ROOT / "train" / "LR"
TRAIN_HR_DIR = DATA_ROOT / "train" / "HR"

TEST_LR_DIR = DATA_ROOT / "test" / "LR"
TEST_HR_DIR = DATA_ROOT / "test" / "HR"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "real"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAPPING_FILE = (
    OUTPUT_DIR
    / "sen2naip_spectral_mapping.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

SCALE = 4
CHANNELS = 4

BAND_NAMES = [
    "B04_Red",
    "B03_Green",
    "B02_Blue",
    "B08_NIR",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FIT AFFINE MAPPING
# ============================================================

def fit_spectral_mapping():

    lr_files = sorted(
        TRAIN_LR_DIR.glob("*.npy")
    )

    hr_files = sorted(
        TRAIN_HR_DIR.glob("*.npy")
    )

    if not lr_files:
        raise RuntimeError(
            "No training LR files found."
        )

    if len(lr_files) != len(hr_files):
        raise RuntimeError(
            "Training LR/HR count mismatch."
        )

    print("\n" + "=" * 70)
    print("FITTING SPECTRAL HARMONIZATION")
    print("=" * 70)

    print(
        "\nTraining pairs:",
        len(lr_files)
    )

    # --------------------------------------------------------
    # Sufficient statistics for:
    #
    # y = a*x + b
    #
    # We avoid loading the entire training set into RAM.
    # --------------------------------------------------------

    n = np.zeros(CHANNELS, dtype=np.float64)

    sum_x = np.zeros(CHANNELS, dtype=np.float64)
    sum_y = np.zeros(CHANNELS, dtype=np.float64)

    sum_xx = np.zeros(CHANNELS, dtype=np.float64)
    sum_xy = np.zeros(CHANNELS, dtype=np.float64)

    # --------------------------------------------------------
    # Process training ROIs
    # --------------------------------------------------------

    for index, lr_path in enumerate(lr_files):

        hr_path = TRAIN_HR_DIR / lr_path.name

        lr = np.load(
            lr_path
        ).astype(np.float32)

        hr = np.load(
            hr_path
        ).astype(np.float32)

        # ----------------------------------------------------
        # Expected:
        #
        # LR = 121 × 121 × 4
        # HR = 484 × 484 × 4
        # ----------------------------------------------------

        if lr.shape[2] != CHANNELS:
            raise RuntimeError(
                f"Unexpected LR shape: {lr.shape}"
            )

        if hr.shape[2] != CHANNELS:
            raise RuntimeError(
                f"Unexpected HR shape: {hr.shape}"
            )

        # ----------------------------------------------------
        # Convert 2.5m HR to 10m reference by averaging
        # each aligned 4×4 block.
        #
        # 484 = 121 × 4
        # ----------------------------------------------------

        h_lr, w_lr, _ = lr.shape

        hr_10m = hr.reshape(
            h_lr,
            SCALE,
            w_lr,
            SCALE,
            CHANNELS,
        ).mean(axis=(1, 3))

        # ----------------------------------------------------
        # Accumulate regression statistics per band
        # ----------------------------------------------------

        x = lr.reshape(-1, CHANNELS).astype(
            np.float64
        )

        y = hr_10m.reshape(-1, CHANNELS).astype(
            np.float64
        )

        n += x.shape[0]

        sum_x += x.sum(axis=0)
        sum_y += y.sum(axis=0)

        sum_xx += (x * x).sum(axis=0)
        sum_xy += (x * y).sum(axis=0)

        if (
            index == 0
            or (index + 1) % 250 == 0
            or index + 1 == len(lr_files)
        ):

            print(
                f"{index + 1}/{len(lr_files)} "
                f"training ROIs processed"
            )

    # --------------------------------------------------------
    # Solve affine regression
    # --------------------------------------------------------

    slopes = np.zeros(CHANNELS)
    intercepts = np.zeros(CHANNELS)

    print("\nLearned mappings:")

    for band in range(CHANNELS):

        denominator = (
            n[band] * sum_xx[band]
            - sum_x[band] ** 2
        )

        if abs(denominator) < 1e-12:
            raise RuntimeError(
                f"Degenerate regression for band {band}"
            )

        slope = (
            n[band] * sum_xy[band]
            - sum_x[band] * sum_y[band]
        ) / denominator

        intercept = (
            sum_y[band]
            - slope * sum_x[band]
        ) / n[band]

        slopes[band] = slope
        intercepts[band] = intercept

        print(
            f"  {BAND_NAMES[band]:8s}: "
            f"y = {slope:.6f} * x "
            f"+ {intercept:.6f}"
        )

    # --------------------------------------------------------
    # Save mapping
    # --------------------------------------------------------

    mapping = {
        "bands": BAND_NAMES,
        "scale": SCALE,
        "slopes": slopes.tolist(),
        "intercepts": intercepts.tolist(),
        "description": (
            "Affine per-band mapping learned exclusively "
            "from SEN2NAIP training ROIs. HR was averaged "
            "from 2.5m to 10m before regression."
        ),
    }

    with open(
        MAPPING_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            mapping,
            f,
            indent=2,
        )

    print(
        "\nMapping saved:",
        MAPPING_FILE
    )

    return slopes, intercepts


# ============================================================
# APPLY SPECTRAL MAPPING
# ============================================================

def apply_mapping(
    lr,
    slopes,
    intercepts,
):
    """
    Apply per-band affine transformation.
    """

    mapped = (
        lr * slopes.reshape(1, 1, CHANNELS)
        + intercepts.reshape(1, 1, CHANNELS)
    )

    return np.clip(
        mapped,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    slopes,
    intercepts,
):

    lr_files = sorted(
        TEST_LR_DIR.glob("*.npy")
    )

    if not lr_files:
        raise RuntimeError(
            "No test LR files found."
        )

    raw_psnr = []
    raw_ssim = []

    harmonized_psnr = []
    harmonized_ssim = []

    print("\n" + "=" * 70)
    print("EVALUATING HARMONIZED BICUBIC")
    print("=" * 70)

    print(
        "\nTest pairs:",
        len(lr_files)
    )

    for index, lr_path in enumerate(lr_files):

        hr_path = TEST_HR_DIR / lr_path.name

        lr = np.load(
            lr_path
        ).astype(np.float32)

        hr = np.load(
            hr_path
        ).astype(np.float32)

        # ----------------------------------------------------
        # Raw LR tensor
        # ----------------------------------------------------

        lr_tensor = torch.from_numpy(
            lr.transpose(2, 0, 1)
        ).unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # RAW BICUBIC
        # ----------------------------------------------------

        with torch.no_grad():

            raw_sr = F.interpolate(
                lr_tensor,
                size=(
                    hr.shape[0],
                    hr.shape[1],
                ),
                mode="bicubic",
                align_corners=False,
            )

        raw_sr = (
            raw_sr
            .squeeze(0)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        raw_sr = np.clip(
            raw_sr,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # HARMONIZE FIRST, THEN BICUBIC
        # ----------------------------------------------------

        mapped_lr = apply_mapping(
            lr,
            slopes,
            intercepts,
        )

        mapped_tensor = torch.from_numpy(
            mapped_lr.transpose(2, 0, 1)
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            harmonized_sr = F.interpolate(
                mapped_tensor,
                size=(
                    hr.shape[0],
                    hr.shape[1],
                ),
                mode="bicubic",
                align_corners=False,
            )

        harmonized_sr = (
            harmonized_sr
            .squeeze(0)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        harmonized_sr = np.clip(
            harmonized_sr,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # Metrics — RAW
        # ----------------------------------------------------

        raw_p = peak_signal_noise_ratio(
            hr,
            raw_sr,
            data_range=1.0,
        )

        raw_s = structural_similarity(
            hr,
            raw_sr,
            channel_axis=2,
            data_range=1.0,
        )

        # ----------------------------------------------------
        # Metrics — HARMONIZED
        # ----------------------------------------------------

        harm_p = peak_signal_noise_ratio(
            hr,
            harmonized_sr,
            data_range=1.0,
        )

        harm_s = structural_similarity(
            hr,
            harmonized_sr,
            channel_axis=2,
            data_range=1.0,
        )

        raw_psnr.append(raw_p)
        raw_ssim.append(raw_s)

        harmonized_psnr.append(harm_p)
        harmonized_ssim.append(harm_s)

        if (
            index < 10
            or (index + 1) % 50 == 0
        ):

            print(
                f"{index + 1:3d}/{len(lr_files)} "
                f"{lr_path.stem}: "
                f"Raw {raw_p:.4f} / {raw_s:.4f} | "
                f"Harmonized {harm_p:.4f} / {harm_s:.4f}"
            )

    # --------------------------------------------------------
    # Calculate final values
    # --------------------------------------------------------

    raw_psnr_mean = np.mean(raw_psnr)
    raw_ssim_mean = np.mean(raw_ssim)

    harm_psnr_mean = np.mean(
        harmonized_psnr
    )

    harm_ssim_mean = np.mean(
        harmonized_ssim
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REAL-DATA BASELINE COMPARISON")
    print("=" * 70)

    print("\nRaw Bicubic ×4")
    print(
        f"PSNR: {raw_psnr_mean:.4f} dB"
    )

    print(
        f"SSIM: {raw_ssim_mean:.4f}"
    )

    print("\nSpectrally Harmonized Bicubic ×4")
    print(
        f"PSNR: {harm_psnr_mean:.4f} dB"
    )

    print(
        f"SSIM: {harm_ssim_mean:.4f}"
    )

    print("\nImprovement from harmonization")

    print(
        f"PSNR: "
        f"{harm_psnr_mean - raw_psnr_mean:+.4f} dB"
    )

    print(
        f"SSIM: "
        f"{harm_ssim_mean - raw_ssim_mean:+.4f}"
    )

    print("\n" + "-" * 70)

    print(
        "Existing CNN:"
    )

    print(
        "PSNR: 22.4025 dB"
    )

    print(
        "SSIM: 0.6458"
    )

    print("-" * 70)

    print("\n✅ Harmonized bicubic benchmark complete.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEN2NAIP SPECTRALLY HARMONIZED BICUBIC BASELINE")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    slopes, intercepts = (
        fit_spectral_mapping()
    )

    evaluate(
        slopes,
        intercepts,
    )


if __name__ == "__main__":
    main()