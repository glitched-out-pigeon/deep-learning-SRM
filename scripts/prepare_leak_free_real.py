import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen2naip"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
    / "leak_free_split.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_clean"
)


# ============================================================
# FIND ZIP
# ============================================================

zip_files = list(
    RAW_ROOT.rglob("*.zip")
)

if not zip_files:
    raise FileNotFoundError(
        f"No SEN2NAIP ZIP found under:\n{RAW_ROOT}"
    )

ZIP_PATH = zip_files[0]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_lr(data: np.ndarray) -> np.ndarray:
    """
    Sentinel-2:
    stored approximately as reflectance * 10000.
    """

    data = (
        data.astype(np.float32)
        / 10000.0
    )

    return np.clip(
        data,
        0.0,
        1.0
    )


def normalize_hr(data: np.ndarray) -> np.ndarray:
    """
    NAIP:
    uint8 [0,255] -> float32 [0,1]
    """

    data = (
        data.astype(np.float32)
        / 255.0
    )

    return np.clip(
        data,
        0.0,
        1.0
    )


# ============================================================
# READ TIFF FROM ZIP
# ============================================================

def read_tiff_from_zip(
    z: zipfile.ZipFile,
    member: str,
) -> np.ndarray:

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = (
            Path(temp_dir)
            / Path(member).name
        )

        with z.open(member) as src:

            with open(
                temp_path,
                "wb"
            ) as dst:

                shutil.copyfileobj(
                    src,
                    dst
                )

        with rasterio.open(temp_path) as src:

            data = src.read()

    return data


# ============================================================
# VALIDATE PAIR
# ============================================================

def validate_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    roi: str,
):

    if lr.ndim != 3:
        raise ValueError(
            f"{roi}: LR is not 3D: {lr.shape}"
        )

    if hr.ndim != 3:
        raise ValueError(
            f"{roi}: HR is not 3D: {hr.shape}"
        )

    # --------------------------------------------------------
    # Four bands
    # --------------------------------------------------------

    if lr.shape[0] != 4:
        raise ValueError(
            f"{roi}: expected 4 LR bands, "
            f"got {lr.shape[0]}"
        )

    if hr.shape[0] != 4:
        raise ValueError(
            f"{roi}: expected 4 HR bands, "
            f"got {hr.shape[0]}"
        )

    # --------------------------------------------------------
    # 4x spatial relationship
    # --------------------------------------------------------

    if hr.shape[1] != lr.shape[1] * 4:

        raise ValueError(
            f"{roi}: height mismatch. "
            f"LR={lr.shape}, HR={hr.shape}"
        )

    if hr.shape[2] != lr.shape[2] * 4:

        raise ValueError(
            f"{roi}: width mismatch. "
            f"LR={lr.shape}, HR={hr.shape}"
        )


# ============================================================
# SAVE PAIR
# ============================================================

def save_pair(
    split: str,
    roi: str,
    lr: np.ndarray,
    hr: np.ndarray,
):

    lr_dir = (
        OUTPUT_ROOT
        / split
        / "LR"
    )

    hr_dir = (
        OUTPUT_ROOT
        / split
        / "HR"
    )

    lr_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    hr_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # C,H,W -> H,W,C
    lr = np.transpose(
        lr,
        (1, 2, 0)
    )

    hr = np.transpose(
        hr,
        (1, 2, 0)
    )

    filename = (
        f"{roi}.npy"
    )

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
    print("PREPARING LEAK-FREE SEN2NAIP DATASET")
    print("=" * 70)

    print("\nZIP:")
    print(ZIP_PATH)

    print("\nSplit definition:")
    print(SPLIT_FILE)

    # --------------------------------------------------------
    # Load split definition
    # --------------------------------------------------------

    if not SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}\n\n"
            "Run rebuild_real_split.py first."
        )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        split_definition = json.load(f)

    splits = {

        "train": split_definition[
            "train_rois"
        ],

        "val": split_definition[
            "val_rois"
        ],

        "test": split_definition[
            "test_rois"
        ],

    }

    # --------------------------------------------------------
    # Basic sanity checks
    # --------------------------------------------------------

    all_rois = []

    for split, rois in splits.items():

        print(
            f"\n{split.upper()} ROIs:",
            len(rois)
        )

        all_rois.extend(rois)

    if len(all_rois) != len(
        set(all_rois)
    ):

        raise RuntimeError(
            "Duplicate ROI exists across split definitions."
        )

    print(
        "\nTotal ROIs:",
        len(all_rois)
    )

    # --------------------------------------------------------
    # Recreate output directory
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous clean dataset:"
        )

        print(
            OUTPUT_ROOT
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    # --------------------------------------------------------
    # Open ZIP
    # --------------------------------------------------------

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

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as z:

        zip_names = set(
            z.namelist()
        )

        # ----------------------------------------------------
        # Process splits
        # ----------------------------------------------------

        for split, roi_list in splits.items():

            print(
                "\n" + "=" * 70
            )

            print(
                f"PROCESSING {split.upper()}"
            )

            print(
                "=" * 70
            )

            total = len(
                roi_list
            )

            for index, roi in enumerate(
                roi_list,
                start=1
            ):

                folder = (
                    f"cross-sensor/{roi}"
                )

                lr_member = (
                    f"{folder}/lr.tif"
                )

                hr_member = (
                    f"{folder}/hr.tif"
                )

                metadata_member = (
                    f"{folder}/metadata.json"
                )

                try:

                    # ------------------------------------------------
                    # Verify archive entries
                    # ------------------------------------------------

                    for member in [
                        lr_member,
                        hr_member,
                        metadata_member,
                    ]:

                        if member not in zip_names:

                            raise FileNotFoundError(
                                f"Missing ZIP entry: "
                                f"{member}"
                            )

                    # ------------------------------------------------
                    # Read imagery
                    # ------------------------------------------------

                    lr = read_tiff_from_zip(
                        z,
                        lr_member
                    )

                    hr = read_tiff_from_zip(
                        z,
                        hr_member
                    )

                    # ------------------------------------------------
                    # Validate
                    # ------------------------------------------------

                    validate_pair(
                        lr,
                        hr,
                        roi,
                    )

                    # ------------------------------------------------
                    # Normalize
                    # ------------------------------------------------

                    lr = normalize_lr(
                        lr
                    )

                    hr = normalize_hr(
                        hr
                    )

                    # ------------------------------------------------
                    # Save
                    # ------------------------------------------------

                    save_pair(
                        split,
                        roi,
                        lr,
                        hr,
                    )

                    counts[split] += 1

                    # ------------------------------------------------
                    # Progress
                    # ------------------------------------------------

                    if (
                        index == 1
                        or index % 100 == 0
                        or index == total
                    ):

                        print(
                            f"{split}: "
                            f"{index}/{total}"
                        )

                except Exception as exc:

                    skipped[split] += 1

                    print(
                        f"\nWARNING: "
                        f"Skipping {roi}"
                    )

                    print(
                        f"Reason: {exc}"
                    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "LEAK-FREE DATASET CREATED"
    )

    print(
        "=" * 70
    )

    print(
        "\nProcessed:"
    )

    print(
        "Train:",
        counts["train"]
    )

    print(
        "Val:",
        counts["val"]
    )

    print(
        "Test:",
        counts["test"]
    )

    print(
        "\nSkipped:"
    )

    print(
        "Train:",
        skipped["train"]
    )

    print(
        "Val:",
        skipped["val"]
    )

    print(
        "Test:",
        skipped["test"]
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nData format:"
    )

    print(
        "LR = 121 × 121 × 4"
    )

    print(
        "HR = 484 × 484 × 4"
    )

    print(
        "LR = 10 m"
    )

    print(
        "HR = 2.5 m"
    )

    print(
        "Scale = 4×"
    )

    print(
        "\n✅ Leak-free SEN2NAIP dataset preparation complete."
    )


if __name__ == "__main__":
    main()