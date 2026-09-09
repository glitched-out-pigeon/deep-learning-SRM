from pathlib import Path
import random
import shutil
import zipfile
import tempfile

import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "sen2naip"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "real"


# ============================================================
# FIND SEN2NAIP ZIP
# ============================================================

ZIP_FILES = list(RAW_ROOT.rglob("*.zip"))

if not ZIP_FILES:
    raise FileNotFoundError(
        f"No SEN2NAIP ZIP file found under:\n{RAW_ROOT}\n\n"
        "Expected something like:\n"
        "data/raw/sen2naip/cross-sensor/cross-sensor.zip"
    )

ZIP_PATH = ZIP_FILES[0]


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_lr(data: np.ndarray) -> np.ndarray:
    """
    Sentinel-2 LR data is stored as scaled integer values.
    Convert approximately to [0, 1] reflectance.
    """
    data = data.astype(np.float32) / 10000.0
    return np.clip(data, 0.0, 1.0)


def normalize_hr(data: np.ndarray) -> np.ndarray:
    """
    NAIP HR data is stored as uint8 [0, 255].
    Convert to [0, 1].
    """
    data = data.astype(np.float32) / 255.0
    return np.clip(data, 0.0, 1.0)


# ============================================================
# READ TIFF FROM ZIP
# ============================================================

def read_tiff_from_zip(
    z: zipfile.ZipFile,
    member: str,
) -> np.ndarray:
    """
    Extract one TIFF temporarily and read it with rasterio.
    """

    with tempfile.TemporaryDirectory() as tmp:

        temp_path = Path(tmp) / Path(member).name

        with z.open(member) as src:
            with open(temp_path, "wb") as dst:
                dst.write(src.read())

        with rasterio.open(temp_path) as src:
            data = src.read()

    return data


# ============================================================
# VALIDATE LR / HR PAIR
# ============================================================

def validate_pair(
    lr: np.ndarray,
    hr: np.ndarray,
) -> None:

    # Both should be [channels, height, width]
    if lr.ndim != 3 or hr.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays. "
            f"Got LR={lr.shape}, HR={hr.shape}"
        )

    # SEN2NAIP cross-sensor data should contain 4 bands
    if lr.shape[0] != 4:
        raise ValueError(
            f"Expected 4 LR bands, got {lr.shape[0]}"
        )

    if hr.shape[0] != 4:
        raise ValueError(
            f"Expected 4 HR bands, got {hr.shape[0]}"
        )

    # HR should be exactly 4x LR spatially
    if hr.shape[1] != lr.shape[1] * 4:
        raise ValueError(
            f"Height mismatch: "
            f"LR={lr.shape}, HR={hr.shape}"
        )

    if hr.shape[2] != lr.shape[2] * 4:
        raise ValueError(
            f"Width mismatch: "
            f"LR={lr.shape}, HR={hr.shape}"
        )


# ============================================================
# SAVE PROCESSED PAIR
# ============================================================

def save_pair(
    split: str,
    roi_name: str,
    lr: np.ndarray,
    hr: np.ndarray,
) -> None:

    lr_dir = OUTPUT_ROOT / split / "LR"
    hr_dir = OUTPUT_ROOT / split / "HR"

    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{roi_name}.npy"

    np.save(
        lr_dir / filename,
        lr.astype(np.float32)
    )

    np.save(
        hr_dir / filename,
        hr.astype(np.float32)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEN2NAIP REAL DATASET PREPARATION")
    print("=" * 70)

    print("\nZIP:")
    print(ZIP_PATH)

    print(
        f"\nZIP size: "
        f"{ZIP_PATH.stat().st_size / (1024 ** 3):.2f} GB"
    )

    # --------------------------------------------------------
    # OPEN ZIP
    # --------------------------------------------------------

    with zipfile.ZipFile(ZIP_PATH, "r") as z:

        names = z.namelist()

        print(f"\nTotal ZIP entries: {len(names)}")

        # Convert to set for fast lookup
        name_set = set(names)

        # ----------------------------------------------------
        # FIND VALID ROI PAIRS
        #
        # IMPORTANT:
        # ZIP paths always use "/" even on Windows.
        # Do not use Windows Path joining here.
        # ----------------------------------------------------

        roi_ids = []

        for name in names:

            # Only inspect LR TIFF files
            if not name.startswith("cross-sensor/"):
                continue

            if not name.endswith("/lr.tif"):
                continue

            # Example:
            # cross-sensor/ROI_0000/lr.tif
            #
            # Get:
            # cross-sensor/ROI_0000

            roi_path = name.rsplit("/", 1)[0]

            hr_path = f"{roi_path}/hr.tif"
            metadata_path = f"{roi_path}/metadata.json"

            # Require all three files
            if (
                hr_path in name_set
                and metadata_path in name_set
            ):
                roi_id = roi_path.split("/")[-1]
                roi_ids.append(roi_id)

        # Remove duplicates and sort
        roi_ids = sorted(set(roi_ids))

        print(f"\nValid paired ROIs: {len(roi_ids)}")

        if len(roi_ids) == 0:
            raise RuntimeError(
                "No valid SEN2NAIP ROI pairs found.\n\n"
                "Expected entries such as:\n"
                "cross-sensor/ROI_0000/lr.tif\n"
                "cross-sensor/ROI_0000/hr.tif\n"
                "cross-sensor/ROI_0000/metadata.json"
            )

        # ----------------------------------------------------
        # ROI-LEVEL SPLIT
        # ----------------------------------------------------

        random.seed(SEED)

        shuffled_ids = roi_ids.copy()
        random.shuffle(shuffled_ids)

        n_total = len(shuffled_ids)

        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        train_ids = shuffled_ids[:n_train]

        val_ids = shuffled_ids[
            n_train:n_train + n_val
        ]

        test_ids = shuffled_ids[
            n_train + n_val:
        ]

        splits = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        print("\n" + "=" * 70)
        print("ROI-LEVEL SPLIT")
        print("=" * 70)

        print(f"\nTotal: {n_total}")
        print(f"Train: {len(train_ids)}")
        print(f"Val:   {len(val_ids)}")
        print(f"Test:  {len(test_ids)}")

        print("\nSplit percentages:")
        print(f"Train: {TRAIN_RATIO * 100:.1f}%")
        print(f"Val:   {VAL_RATIO * 100:.1f}%")
        print(f"Test:  {TEST_RATIO * 100:.1f}%")

        # ----------------------------------------------------
        # CLEAR OLD PROCESSED DATA
        # ----------------------------------------------------

        if OUTPUT_ROOT.exists():

            print(
                "\nRemoving previous processed real dataset..."
            )

            shutil.rmtree(OUTPUT_ROOT)

        # ----------------------------------------------------
        # PROCESS EACH SPLIT
        # ----------------------------------------------------

        counts = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        skipped = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        for split, roi_list in splits.items():

            print("\n" + "=" * 70)
            print(f"PROCESSING {split.upper()}")
            print("=" * 70)

            total = len(roi_list)

            for index, roi in enumerate(roi_list, start=1):

                folder = f"cross-sensor/{roi}"

                lr_member = f"{folder}/lr.tif"
                hr_member = f"{folder}/hr.tif"

                try:

                    # ----------------------------------------
                    # READ
                    # ----------------------------------------

                    lr = read_tiff_from_zip(
                        z,
                        lr_member
                    )

                    hr = read_tiff_from_zip(
                        z,
                        hr_member
                    )

                    # ----------------------------------------
                    # VALIDATE
                    # ----------------------------------------

                    validate_pair(lr, hr)

                    # ----------------------------------------
                    # NORMALIZE
                    # ----------------------------------------

                    lr = normalize_lr(lr)
                    hr = normalize_hr(hr)

                    # ----------------------------------------
                    # CHANNEL-FIRST → CHANNEL-LAST
                    #
                    # Current:
                    # C,H,W
                    #
                    # Saved:
                    # H,W,C
                    # ----------------------------------------

                    lr = np.transpose(
                        lr,
                        (1, 2, 0)
                    )

                    hr = np.transpose(
                        hr,
                        (1, 2, 0)
                    )

                    # ----------------------------------------
                    # SAVE
                    # ----------------------------------------

                    save_pair(
                        split,
                        roi,
                        lr,
                        hr
                    )

                    counts[split] += 1

                    # Progress
                    if (
                        index == 1
                        or index % 100 == 0
                        or index == total
                    ):
                        print(
                            f"{split}: "
                            f"{index}/{total} "
                            f"processed"
                        )

                except Exception as exc:

                    skipped[split] += 1

                    print(
                        f"\nWARNING: Skipping {roi}"
                    )

                    print(
                        f"Reason: {exc}"
                    )

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("REAL DATASET CREATED")
        print("=" * 70)

        print(f"\nTrain pairs: {counts['train']}")
        print(f"Val pairs:   {counts['val']}")
        print(f"Test pairs:  {counts['test']}")

        print("\nSkipped:")
        print(f"Train: {skipped['train']}")
        print(f"Val:   {skipped['val']}")
        print(f"Test:  {skipped['test']}")

        print("\nOutput:")
        print(OUTPUT_ROOT)

        print("\n" + "-" * 70)
        print("DATA FORMAT")
        print("-" * 70)

        print("LR:")
        print("  Bands: 4")
        print("  Resolution: 10 m")
        print("  Normalization: /10000")
        print("  Stored shape: H x W x 4")

        print("\nHR:")
        print("  Bands: 4")
        print("  Resolution: 2.5 m")
        print("  Normalization: /255")
        print("  Stored shape: H x W x 4")

        print("\n" + "-" * 70)
        print("SUPER-RESOLUTION SCALE")
        print("-" * 70)

        print("10 m → 2.5 m")
        print("Scale factor: 4x")

        print("\n" + "-" * 70)
        print("DATA SPLIT")
        print("-" * 70)

        print("Split performed at ROI level.")
        print("No ROI is intentionally placed in multiple splits.")

        print("\n✅ SEN2NAIP preparation complete.")


if __name__ == "__main__":
    main()