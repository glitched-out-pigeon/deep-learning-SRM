from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
    / "test"
    / "LR"
)

HR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
    / "test"
    / "HR"
)


# ============================================================
# TEST ONE REAL PAIR
# ============================================================

lr_path = sorted(LR_DIR.glob("*.npy"))[0]
hr_path = HR_DIR / lr_path.name

lr = np.load(lr_path)
hr = np.load(hr_path)


print("=" * 70)
print("REAL SEN2NAIP BAND INSPECTION")
print("=" * 70)

print("\nPair:")
print(lr_path.name)

print("\nLR shape:", lr.shape)
print("HR shape:", hr.shape)

print("\nExpected Sentinel-2 bands:")
print("Channel 0 = B04 (Red)")
print("Channel 1 = B03 (Green)")
print("Channel 2 = B02 (Blue)")
print("Channel 3 = B08 (NIR)")

print("\nPer-band statistics")
print("-" * 70)

band_names = [
    "B04 Red",
    "B03 Green",
    "B02 Blue",
    "B08 NIR",
]

for i, name in enumerate(band_names):

    lr_band = lr[:, :, i]
    hr_band = hr[:, :, i]

    print(f"\n{name}")

    print(
        f"  LR: "
        f"min={lr_band.min():.6f}, "
        f"max={lr_band.max():.6f}, "
        f"mean={lr_band.mean():.6f}, "
        f"std={lr_band.std():.6f}"
    )

    print(
        f"  HR: "
        f"min={hr_band.min():.6f}, "
        f"max={hr_band.max():.6f}, "
        f"mean={hr_band.mean():.6f}, "
        f"std={hr_band.std():.6f}"
    )

print("\n" + "=" * 70)
print("✅ BAND INSPECTION COMPLETE")
print("=" * 70)