from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "real"


# ============================================================
# HELPERS
# ============================================================

def inspect_split(split: str):
    lr_dir = DATA_ROOT / split / "LR"
    hr_dir = DATA_ROOT / split / "HR"

    lr_files = sorted(lr_dir.glob("*.npy"))
    hr_files = sorted(hr_dir.glob("*.npy"))

    print("\n" + "=" * 70)
    print(f"{split.upper()} DATASET")
    print("=" * 70)

    print(f"LR files: {len(lr_files)}")
    print(f"HR files: {len(hr_files)}")

    if len(lr_files) != len(hr_files):
        raise RuntimeError(
            f"LR/HR file count mismatch in {split}: "
            f"{len(lr_files)} vs {len(hr_files)}"
        )

    if not lr_files:
        raise RuntimeError(
            f"No files found in {split}"
        )

    # Check filename pairing
    lr_names = {f.name for f in lr_files}
    hr_names = {f.name for f in hr_files}

    if lr_names != hr_names:
        missing_hr = sorted(lr_names - hr_names)
        missing_lr = sorted(hr_names - lr_names)

        raise RuntimeError(
            f"Pair mismatch in {split}\n"
            f"Missing HR: {missing_hr[:5]}\n"
            f"Missing LR: {missing_lr[:5]}"
        )

    # Inspect first pair
    filename = lr_files[0].name

    lr = np.load(lr_dir / filename)
    hr = np.load(hr_dir / filename)

    print("\nExample:")
    print(f"  {filename}")

    print("\nLR:")
    print(f"  Shape: {lr.shape}")
    print(f"  Dtype: {lr.dtype}")
    print(f"  Min:   {lr.min():.6f}")
    print(f"  Max:   {lr.max():.6f}")
    print(f"  Mean:  {lr.mean():.6f}")

    print("\nHR:")
    print(f"  Shape: {hr.shape}")
    print(f"  Dtype: {hr.dtype}")
    print(f"  Min:   {hr.min():.6f}")
    print(f"  Max:   {hr.max():.6f}")
    print(f"  Mean:  {hr.mean():.6f}")

    # Shape checks
    if lr.shape != (121, 121, 4):
        raise RuntimeError(
            f"Unexpected LR shape: {lr.shape}"
        )

    if hr.shape != (484, 484, 4):
        raise RuntimeError(
            f"Unexpected HR shape: {hr.shape}"
        )

    # Numeric checks
    if lr.dtype != np.float32:
        raise RuntimeError(
            f"LR dtype should be float32, got {lr.dtype}"
        )

    if hr.dtype != np.float32:
        raise RuntimeError(
            f"HR dtype should be float32, got {hr.dtype}"
        )

    if lr.min() < 0.0 or lr.max() > 1.0:
        raise RuntimeError(
            "LR values are outside [0, 1]"
        )

    if hr.min() < 0.0 or hr.max() > 1.0:
        raise RuntimeError(
            "HR values are outside [0, 1]"
        )

    # Check 4x spatial relationship
    if hr.shape[0] != lr.shape[0] * 4:
        raise RuntimeError("Height is not 4x.")

    if hr.shape[1] != lr.shape[1] * 4:
        raise RuntimeError("Width is not 4x.")

    print("\n✅ Shape, dtype, range and pairing checks passed.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEN2NAIP PROCESSED DATASET TEST")
    print("=" * 70)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATA_ROOT}"
        )

    inspect_split("train")
    inspect_split("val")
    inspect_split("test")

    print("\n" + "=" * 70)
    print("✅ REAL DATASET VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()