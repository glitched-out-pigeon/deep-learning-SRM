from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "highres"
    / "bengaluru_20260427"
)

REFERENCE = SCENE_DIR / "B02.tiff"
B08_INPUT = SCENE_DIR / "B08.tiff"
B08_OUTPUT = SCENE_DIR / "B08_aligned.tiff"


# ============================================================
# CHECK
# ============================================================

if not REFERENCE.exists():
    raise FileNotFoundError(
        f"Reference band not found:\n{REFERENCE}"
    )

if not B08_INPUT.exists():
    raise FileNotFoundError(
        f"B08 not found:\n{B08_INPUT}"
    )


# ============================================================
# LOAD REFERENCE GRID
# ============================================================

with rasterio.open(REFERENCE) as ref:

    reference_data = ref.read(1)

    reference_profile = ref.profile.copy()

    reference_width = ref.width
    reference_height = ref.height

    reference_crs = ref.crs
    reference_transform = ref.transform

    reference_dtype = reference_data.dtype


# ============================================================
# LOAD B08
# ============================================================

with rasterio.open(B08_INPUT) as src:

    b08 = src.read(1).astype(
        np.float32
    )

    source_crs = src.crs
    source_transform = src.transform

    source_width = src.width
    source_height = src.height


# ============================================================
# PRINT INFORMATION
# ============================================================

print("=" * 70)
print("ALIGNING B08 TO EXACT B02 GRID")
print("=" * 70)

print("\nReference:")
print("  File:", REFERENCE)
print("  Size:", reference_width, "×", reference_height)
print("  CRS:", reference_crs)
print("  Transform:")
print(reference_transform)

print("\nB08 source:")
print("  File:", B08_INPUT)
print("  Size:", source_width, "×", source_height)
print("  CRS:", source_crs)
print("  Transform:")
print(source_transform)


# ============================================================
# CHECK CRS
# ============================================================

if source_crs != reference_crs:

    raise RuntimeError(
        "B08 and B02 use different CRS. "
        "This script currently expects matching CRS."
    )


# ============================================================
# DESTINATION ARRAY
# ============================================================

aligned = np.zeros(
    (
        reference_height,
        reference_width,
    ),
    dtype=np.float32,
)


# ============================================================
# REPROJECT / RESAMPLE
# ============================================================

reproject(
    source=b08,
    destination=aligned,
    src_transform=source_transform,
    src_crs=source_crs,
    dst_transform=reference_transform,
    dst_crs=reference_crs,
    dst_width=reference_width,
    dst_height=reference_height,
    resampling=Resampling.bilinear,
)


# ============================================================
# OUTPUT PROFILE
# ============================================================

profile = reference_profile.copy()

profile.update(
    driver="GTiff",
    width=reference_width,
    height=reference_height,
    count=1,
    dtype="float32",
    compress="deflate",
)


# ============================================================
# SAVE
# ============================================================

with rasterio.open(
    B08_OUTPUT,
    "w",
    **profile,
) as dst:

    dst.write(
        aligned,
        1,
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("B08 ALIGNED")
print("=" * 70)

print(
    "\nOutput:",
    B08_OUTPUT
)

print(
    "\nShape:",
    aligned.shape
)

print(
    "Range:",
    f"{aligned.min():.6f} -> "
    f"{aligned.max():.6f}"
)

print(
    "\n✅ B08 has been regridded onto the exact B02 grid."
)