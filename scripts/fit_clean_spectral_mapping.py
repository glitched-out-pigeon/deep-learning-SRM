from pathlib import Path
import json

import numpy as np


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

TRAIN_LR_DIR = (
    DATA_ROOT
    / "train"
    / "LR"
)

TRAIN_HR_DIR = (
    DATA_ROOT
    / "train"
    / "HR"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "real_clean"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAPPING_FILE = (
    OUTPUT_DIR
    / "clean_spectral_mapping.json"
)


# ============================================================
# CONFIG
# ============================================================

CHANNELS = 4
SCALE = 4

BAND_NAMES = [
    "B04_Red",
    "B03_Green",
    "B02_Blue",
    "B08_NIR",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLEAN SEN2NAIP SPECTRAL MAPPING")
    print("=" * 70)

    lr_files = sorted(
        TRAIN_LR_DIR.glob("*.npy")
    )

    if not lr_files:
        raise RuntimeError(
            f"No training files found:\n{TRAIN_LR_DIR}"
        )

    print(
        "\nTraining ROIs:",
        len(lr_files)
    )

    # --------------------------------------------------------
    # Regression statistics
    #
    # y = a*x + b
    #
    # x = Sentinel-2 value
    # y = NAIP value averaged to 10m
    # --------------------------------------------------------

    n = np.zeros(
        CHANNELS,
        dtype=np.float64,
    )

    sum_x = np.zeros(
        CHANNELS,
        dtype=np.float64,
    )

    sum_y = np.zeros(
        CHANNELS,
        dtype=np.float64,
    )

    sum_xx = np.zeros(
        CHANNELS,
        dtype=np.float64,
    )

    sum_xy = np.zeros(
        CHANNELS,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Process training data
    # --------------------------------------------------------

    for index, lr_path in enumerate(
        lr_files,
        start=1,
    ):

        hr_path = (
            TRAIN_HR_DIR
            / lr_path.name
        )

        lr = np.load(
            lr_path
        ).astype(np.float32)

        hr = np.load(
            hr_path
        ).astype(np.float32)

        if lr.shape != (
            121,
            121,
            4,
        ):

            raise RuntimeError(
                f"Unexpected LR shape: "
                f"{lr.shape}"
            )

        if hr.shape != (
            484,
            484,
            4,
        ):

            raise RuntimeError(
                f"Unexpected HR shape: "
                f"{hr.shape}"
            )

        # ----------------------------------------------------
        # Convert NAIP 2.5m to 10m
        # ----------------------------------------------------

        hr_10m = hr.reshape(
            121,
            4,
            121,
            4,
            4,
        ).mean(
            axis=(1, 3)
        )

        x = lr.reshape(
            -1,
            CHANNELS,
        ).astype(
            np.float64
        )

        y = hr_10m.reshape(
            -1,
            CHANNELS,
        ).astype(
            np.float64
        )

        n += x.shape[0]
        sum_x += x.sum(axis=0)
        sum_y += y.sum(axis=0)
        sum_xx += (x * x).sum(axis=0)
        sum_xy += (x * y).sum(axis=0)

        if (
            index == 1
            or index % 250 == 0
            or index == len(lr_files)
        ):

            print(
                f"{index}/{len(lr_files)} processed"
            )

    # --------------------------------------------------------
    # Solve regressions
    # --------------------------------------------------------

    slopes = np.zeros(
        CHANNELS,
        dtype=np.float32,
    )

    intercepts = np.zeros(
        CHANNELS,
        dtype=np.float32,
    )

    print("\n" + "=" * 70)
    print("LEARNED CLEAN MAPPING")
    print("=" * 70)

    for band in range(CHANNELS):

        denominator = (
            n[band] * sum_xx[band]
            - sum_x[band] ** 2
        )

        if abs(denominator) < 1e-12:

            raise RuntimeError(
                f"Degenerate regression "
                f"for channel {band}"
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
            f"{BAND_NAMES[band]:8s}: "
            f"y = {slope:.6f}x "
            f"+ {intercept:.6f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = {
        "bands": BAND_NAMES,
        "scale": SCALE,
        "slopes": slopes.tolist(),
        "intercepts": intercepts.tolist(),
        "training_rois": len(lr_files),
        "description": (
            "Spectral affine mapping fitted exclusively "
            "on the leak-free training split. "
            "NAIP HR was averaged to the Sentinel-2 "
            "10m grid before fitting."
        ),
    }

    with open(
        MAPPING_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print("\nSaved:")
    print(MAPPING_FILE)

    print(
        "\n✅ Clean spectral mapping complete."
    )


if __name__ == "__main__":
    main()