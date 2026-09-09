import sys
from pathlib import Path
import json

import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt


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

SCENE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "highres"
    / "bengaluru_20260427"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bengaluru_inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "real_clean"
    / "sen2naip_clean_harmonized_best.pth"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "real_clean"
    / "clean_spectral_mapping.json"
)

OUTPUT_TIF = (
    OUTPUT_DIR
    / "bengaluru_cnn_sr_2p5m.tif"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "bengaluru_genuine_before_after.png"
)


# ============================================================
# CONFIG
# ============================================================

TILE_SIZE = 64
OVERLAP = 16
STRIDE = TILE_SIZE - OVERLAP
SCALE = 4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# INPUT BANDS
# ============================================================

BAND_FILES = {
    "B04": SCENE_DIR / "B04.tiff",
    "B03": SCENE_DIR / "B03.tiff",
    "B02": SCENE_DIR / "B02.tiff",
    "B08": SCENE_DIR / "B08_aligned.tiff",
}


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("GENUINE BENGALURU SR INFERENCE")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

for band, path in BAND_FILES.items():

    if not path.exists():

        raise FileNotFoundError(
            f"Missing {band}:\n{path}"
        )


# ============================================================
# LOAD CLEAN SPECTRAL MAPPING
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
# LOAD CLEAN MODEL
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
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "\nModel checkpoint epoch:",
    checkpoint.get("epoch")
)

print(
    "Model validation loss:",
    checkpoint.get("val_loss")
)


# ============================================================
# LOAD RAW SENTINEL-2
# ============================================================

arrays = []

reference_profile = None
reference_crs = None
reference_transform = None
reference_width = None
reference_height = None


for band in [
    "B04",
    "B03",
    "B02",
    "B08",
]:

    path = BAND_FILES[band]

    with rasterio.open(path) as src:

        data = src.read(1).astype(
            np.float32
        )

        if reference_profile is None:

            reference_profile = (
                src.profile.copy()
            )

            reference_crs = src.crs
            reference_transform = src.transform
            reference_width = src.width
            reference_height = src.height

        else:

            if (
                src.width != reference_width
                or src.height != reference_height
            ):

                raise RuntimeError(
                    f"{band} dimensions do not match."
                )

            if src.crs != reference_crs:

                raise RuntimeError(
                    f"{band} CRS does not match."
                )

            if src.transform != reference_transform:

                raise RuntimeError(
                    f"{band} geotransform does not match."
                )

        arrays.append(data)


# B04, B03, B02, B08
scene = np.stack(
    arrays,
    axis=-1
).astype(np.float32)


# ============================================================
# IMPORTANT:
# USE THE ACTUAL VALUES AS STORED IN THE INPUT FILES
# ============================================================

print("\nRAW SENTINEL-2")

print(
    "Shape:",
    scene.shape
)

print(
    "CRS:",
    reference_crs
)

print(
    "Resolution:",
    (
        abs(reference_transform.a),
        abs(reference_transform.e)
    )
)

print(
    "Range:",
    f"{scene.min():.6f} -> "
    f"{scene.max():.6f}"
)


# ============================================================
# INPUT PREPARATION — EXACT SAME LOGIC AS MODEL TRAINING
# ============================================================

# The Bengaluru TIFFs were already stored as reflectance-like
# floating point values, so no /10000 conversion is performed.

scene = np.clip(
    scene,
    0.0,
    1.0
).astype(np.float32)


# ============================================================
# SAME CLEAN SPECTRAL HARMONIZATION USED DURING TRAINING
# ============================================================

harmonized = (
    scene
    * slopes.reshape(1, 1, 4)
    + intercepts.reshape(1, 1, 4)
)

harmonized = np.clip(
    harmonized,
    0.0,
    1.0
).astype(np.float32)


print(
    "\nMODEL INPUT AFTER CLEAN HARMONIZATION"
)

print(
    "Shape:",
    harmonized.shape
)

print(
    "Range:",
    f"{harmonized.min():.6f} -> "
    f"{harmonized.max():.6f}"
)


# ============================================================
# OUTPUT SIZE
# ============================================================

height, width, channels = (
    harmonized.shape
)

out_height = height * SCALE
out_width = width * SCALE


print(
    "\nExpected SR output:",
    f"{out_width} × {out_height}"
)


# ============================================================
# ACCUMULATION BUFFERS
# ============================================================

sr_sum = np.zeros(
    (
        out_height,
        out_width,
        4
    ),
    dtype=np.float32
)

weight_sum = np.zeros(
    (
        out_height,
        out_width,
        1
    ),
    dtype=np.float32
)


# ============================================================
# TILE POSITIONS
# ============================================================

def positions(length, tile, stride):

    values = list(
        range(
            0,
            max(
                1,
                length - tile + 1
            ),
            stride
        )
    )

    last = length - tile

    if last > 0:

        if values[-1] != last:

            values.append(last)

    return values


y_positions = positions(
    height,
    TILE_SIZE,
    STRIDE
)

x_positions = positions(
    width,
    TILE_SIZE,
    STRIDE
)


# ============================================================
# BLENDING WINDOW
# ============================================================

window_1d = np.hanning(
    TILE_SIZE
).astype(np.float32)

window_2d = (
    window_1d[:, None]
    * window_1d[None, :]
)

# Prevent zero-weight borders
window_2d = np.maximum(
    window_2d,
    0.05
)

window_hr = np.repeat(
    np.repeat(
        window_2d,
        SCALE,
        axis=0
    ),
    SCALE,
    axis=1
)

window_hr = window_hr[:, :, None]


# ============================================================
# CNN INFERENCE
# ============================================================

total_tiles = (
    len(y_positions)
    * len(x_positions)
)

tile_number = 0


print("\n" + "=" * 70)
print("CNN INFERENCE")
print("=" * 70)


with torch.no_grad():

    for y in y_positions:

        for x in x_positions:

            tile_number += 1

            # Exact model input tile
            tile = harmonized[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE,
                :
            ]

            tensor = torch.from_numpy(
                tile.transpose(2, 0, 1)
            ).unsqueeze(0).to(
                DEVICE
            )

            # Exact CNN prediction
            prediction = model(
                tensor
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
                0.0,
                1.0
            )

            oy = y * SCALE
            ox = x * SCALE

            sr_sum[
                oy:oy + TILE_SIZE * SCALE,
                ox:ox + TILE_SIZE * SCALE,
                :
            ] += (
                prediction
                * window_hr
            )

            weight_sum[
                oy:oy + TILE_SIZE * SCALE,
                ox:ox + TILE_SIZE * SCALE,
                :
            ] += window_hr

            if (
                tile_number == 1
                or tile_number % 25 == 0
                or tile_number == total_tiles
            ):

                print(
                    f"{tile_number}/{total_tiles}"
                )


# ============================================================
# STITCH TILES
# ============================================================

sr_scene = (
    sr_sum
    / np.maximum(
        weight_sum,
        1e-8
    )
)

sr_scene = np.clip(
    sr_scene,
    0.0,
    1.0
).astype(np.float32)


print(
    "\nSR output range:",
    f"{sr_scene.min():.6f} -> "
    f"{sr_scene.max():.6f}"
)


# ============================================================
# SAVE ACTUAL 2.5m GEOTIFF
# ============================================================

output_profile = (
    reference_profile.copy()
)

output_profile.update(
    driver="GTiff",
    height=out_height,
    width=out_width,
    count=4,
    dtype="float32",
    compress="deflate",
    transform=rasterio.Affine(
        reference_transform.a / SCALE,
        reference_transform.b,
        reference_transform.c,
        reference_transform.d,
        reference_transform.e / SCALE,
        reference_transform.f
    )
)


with rasterio.open(
    OUTPUT_TIF,
    "w",
    **output_profile
) as dst:

    for band_index in range(4):

        dst.write(
            sr_scene[:, :, band_index],
            band_index + 1
        )


print(
    "\nActual SR GeoTIFF saved:"
)

print(
    OUTPUT_TIF
)


# ============================================================
# DISPLAY
# ============================================================

def display_rgb(image):
    """
    Convert B04/B03/B02 to RGB for visualization only.

    Percentile stretching is performed on the RGB channels
    so the satellite imagery is visible on a normal monitor.
    No image data is altered.
    """

    rgb = image[:, :, :3].astype(
        np.float32
    )

    output = np.zeros_like(
        rgb
    )

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
                (
                    channel - low
                )
                / (
                    high - low
                ),
                0,
                1
            )

    return output


# Exact model input displayed at its native resolution.
input_rgb = display_rgb(
    harmonized
)

# Exact CNN output displayed at its native output resolution.
sr_rgb = display_rgb(
    sr_scene
)


# ============================================================
# GENUINE TWO-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 8)
)


# ------------------------------------------------------------
# LEFT — EXACT MODEL INPUT
# ------------------------------------------------------------

axes[0].imshow(
    input_rgb,
    interpolation="nearest"
)

axes[0].set_title(
    "ACTUAL CNN INPUT\n"
    "Sentinel-2 • 10m",
    fontsize=18,
    fontweight="bold"
)


# ------------------------------------------------------------
# RIGHT — EXACT CNN OUTPUT
# ------------------------------------------------------------

axes[1].imshow(
    sr_rgb,
    interpolation="nearest"
)

axes[1].set_title(
    "ACTUAL CNN OUTPUT\n"
    "Estimated 2.5m • 4×",
    fontsize=18,
    fontweight="bold"
)


# ------------------------------------------------------------
# Remove axes
# ------------------------------------------------------------

for ax in axes:

    ax.axis("off")


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Deep Learning Based Super-Resolution Mapping\n"
    "Bengaluru Sentinel-2 → CNN Estimated 2.5m",
    fontsize=21,
    fontweight="bold"
)


plt.tight_layout(
    rect=(0, 0, 1, 0.94)
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_PNG,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("GENUINE COMPARISON COMPLETE")
print("=" * 70)

print(
    "\nComparison:"
)

print(
    OUTPUT_PNG
)

print(
    "\nLEFT PANEL:"
)

print(
    "Exact harmonized Sentinel-2 data "
    "fed to the CNN."
)

print(
    "\nRIGHT PANEL:"
)

print(
    "Exact stitched CNN prediction."
)

print(
    "\nNo blur, sharpening, bicubic, "
    "or synthetic degradation was applied."
)

print(
    "\n✅ Genuine before/after visualization created."
)