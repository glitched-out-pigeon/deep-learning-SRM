import sys
from pathlib import Path
import json
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

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

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen2naip"
)

TRAIN_LR = DATA_ROOT / "train" / "LR"
VAL_LR = DATA_ROOT / "val" / "LR"
TEST_LR = DATA_ROOT / "test" / "LR"

TRAIN_HR = DATA_ROOT / "train" / "HR"
TEST_HR = DATA_ROOT / "test" / "HR"

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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ZIP
# ============================================================

zip_files = list(
    RAW_ROOT.rglob("*.zip")
)

if not zip_files:
    raise FileNotFoundError(
        "SEN2NAIP ZIP not found."
    )

ZIP_PATH = zip_files[0]


# ============================================================
# LOAD MAPPING
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
# BUILD SPLIT MAP
# ============================================================

split_dirs = {
    "train": TRAIN_LR,
    "val": VAL_LR,
    "test": TEST_LR,
}

roi_to_split = {}

for split, directory in split_dirs.items():

    for path in directory.glob("*.npy"):

        roi = path.stem

        if roi in roi_to_split:

            print(
                f"WARNING: ROI appears in multiple splits: {roi}"
            )

        roi_to_split[roi] = split


# ============================================================
# CHECK RAW DATASET FOR SOURCE-LEVEL LEAKAGE
# ============================================================

print("=" * 70)
print("SEN2NAIP SPLIT + RESULT AUDIT")
print("=" * 70)

print("\nZIP:", ZIP_PATH)

source_info = {}

with zipfile.ZipFile(
    ZIP_PATH,
    "r",
) as z:

    names = z.namelist()
    name_set = set(names)

    roi_folders = sorted({
        name.rsplit("/", 1)[0].split("/")[-1]
        for name in names
        if name.startswith("cross-sensor/")
        and name.endswith("/metadata.json")
    })

    print(
        "\nMetadata ROIs found:",
        len(roi_folders)
    )

    for roi in roi_folders:

        metadata_path = (
            f"cross-sensor/{roi}/metadata.json"
        )

        try:

            metadata = json.loads(
                z.read(metadata_path)
                .decode("utf-8")
            )

            source_info[roi] = metadata

        except Exception as exc:

            print(
                f"Could not read {roi}: {exc}"
            )


# ============================================================
# SOURCE-LEVEL DUPLICATION CHECKS
# ============================================================

def find_duplicates(field):

    values = {}

    for roi, metadata in source_info.items():

        value = metadata.get(field)

        if value is None:
            continue

        split = roi_to_split.get(roi)

        if split is None:
            continue

        values.setdefault(
            value,
            []
        ).append(
            (roi, split)
        )

    duplicates = []

    for value, entries in values.items():

        splits = {
            split
            for _, split in entries
        }

        if len(entries) > 1 and len(splits) > 1:

            duplicates.append(
                (value, entries)
            )

    return duplicates


for field in [
    "s2_id",
    "naip_id",
    "roi_id",
    "proj:geometry",
]:

    duplicates = find_duplicates(
        field
    )

    print("\n" + "-" * 70)

    print(
        f"{field} duplicates across splits:"
    )

    if not duplicates:

        print(
            "  NONE FOUND"
        )

    else:

        print(
            f"  {len(duplicates)} duplicated source values"
        )

        for value, entries in duplicates[:10]:

            print(
                "\n ",
                value
            )

            for roi, split in entries:

                print(
                    f"    {roi} -> {split}"
                )


# ============================================================
# MODEL AUDIT ON ALL TEST ROIS
# ============================================================

test_files = sorted(
    TEST_LR.glob("*.npy")
)

print("\n" + "=" * 70)
print("FULL TEST RESULT AUDIT")
print("=" * 70)

results = []


for index, lr_path in enumerate(test_files):

    hr_path = TEST_HR / lr_path.name

    lr = np.load(
        lr_path
    ).astype(np.float32)

    hr = np.load(
        hr_path
    ).astype(np.float32)

    # --------------------------------------------------------
    # Harmonization
    # --------------------------------------------------------

    harmonized = (
        lr
        * slopes.reshape(1, 1, 4)
        + intercepts.reshape(1, 1, 4)
    )

    harmonized = np.clip(
        harmonized,
        0.0,
        1.0,
    ).astype(np.float32)

    x = torch.from_numpy(
        harmonized.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Bicubic
    # --------------------------------------------------------

    bicubic = F.interpolate(
        x,
        size=(484, 484),
        mode="bicubic",
        align_corners=False,
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    with torch.no_grad():

        sr = model(x)

    bicubic_np = (
        bicubic
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    sr_np = (
        sr
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    bicubic_np = np.clip(
        bicubic_np,
        0,
        1,
    )

    sr_np = np.clip(
        sr_np,
        0,
        1,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    b_psnr = peak_signal_noise_ratio(
        hr,
        bicubic_np,
        data_range=1.0,
    )

    c_psnr = peak_signal_noise_ratio(
        hr,
        sr_np,
        data_range=1.0,
    )

    b_ssim = structural_similarity(
        hr,
        bicubic_np,
        channel_axis=2,
        data_range=1.0,
    )

    c_ssim = structural_similarity(
        hr,
        sr_np,
        channel_axis=2,
        data_range=1.0,
    )

    # --------------------------------------------------------
    # Per-band PSNR
    # --------------------------------------------------------

    band_b_psnr = []
    band_c_psnr = []

    for band in range(4):

        band_b = peak_signal_noise_ratio(
            hr[:, :, band],
            bicubic_np[:, :, band],
            data_range=1.0,
        )

        band_c = peak_signal_noise_ratio(
            hr[:, :, band],
            sr_np[:, :, band],
            data_range=1.0,
        )

        band_b_psnr.append(
            band_b
        )

        band_c_psnr.append(
            band_c
        )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    residual = (
        sr_np
        - bicubic_np
    )

    residual_mae = np.mean(
        np.abs(residual)
    )

    residual_rms = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    residual_p95 = np.percentile(
        np.abs(residual),
        95,
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append(
        {
            "name": lr_path.stem,
            "bicubic_psnr": b_psnr,
            "cnn_psnr": c_psnr,
            "psnr_gain": c_psnr - b_psnr,
            "bicubic_ssim": b_ssim,
            "cnn_ssim": c_ssim,
            "ssim_gain": c_ssim - b_ssim,
            "band_b_psnr": band_b_psnr,
            "band_c_psnr": band_c_psnr,
            "residual_mae": residual_mae,
            "residual_rms": residual_rms,
            "residual_p95": residual_p95,
        }
    )


# ============================================================
# SORT
# ============================================================

results_by_gain = sorted(
    results,
    key=lambda x: x["psnr_gain"],
)

best = results_by_gain[-1]
worst = results_by_gain[0]

median = results_by_gain[
    len(results_by_gain) // 2
]


# ============================================================
# OVERALL STATISTICS
# ============================================================

gains = np.asarray([
    r["psnr_gain"]
    for r in results
])

ssim_gains = np.asarray([
    r["ssim_gain"]
    for r in results
])

cnn_psnr = np.asarray([
    r["cnn_psnr"]
    for r in results
])

bicubic_psnr = np.asarray([
    r["bicubic_psnr"]
    for r in results
])


print("\n" + "=" * 70)
print("OVERALL DISTRIBUTION")
print("=" * 70)

print(
    "\nCNN PSNR mean:",
    f"{cnn_psnr.mean():.4f}"
)

print(
    "CNN PSNR std:",
    f"{cnn_psnr.std():.4f}"
)

print(
    "Bicubic PSNR mean:",
    f"{bicubic_psnr.mean():.4f}"
)

print(
    "Bicubic PSNR std:",
    f"{bicubic_psnr.std():.4f}"
)

print(
    "\nPSNR gain mean:",
    f"{gains.mean():+.4f}"
)

print(
    "PSNR gain std:",
    f"{gains.std():.4f}"
)

print(
    "PSNR gain median:",
    f"{np.median(gains):+.4f}"
)

print(
    "Minimum gain:",
    f"{gains.min():+.4f}"
)

print(
    "Maximum gain:",
    f"{gains.max():+.4f}"
)

print(
    "\nROIs where CNN improves PSNR:",
    f"{np.sum(gains > 0)}/{len(gains)} "
    f"({100 * np.mean(gains > 0):.1f}%)"
)

print(
    "ROIs where CNN improves SSIM:",
    f"{np.sum(ssim_gains > 0)}/{len(ssim_gains)} "
    f"({100 * np.mean(ssim_gains > 0):.1f}%)"
)


# ============================================================
# INSPECT TOP / BOTTOM
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 PSNR GAINS")
print("=" * 70)

for r in results_by_gain[-10:][::-1]:

    print(
        f"{r['name']}: "
        f"{r['bicubic_psnr']:.2f} → "
        f"{r['cnn_psnr']:.2f} "
        f"({r['psnr_gain']:+.2f} dB), "
        f"SSIM {r['ssim_gain']:+.4f}, "
        f"residual RMS={r['residual_rms']:.6f}"
    )


print("\n" + "=" * 70)
print("BOTTOM 10 PSNR GAINS")
print("=" * 70)

for r in results_by_gain[:10]:

    print(
        f"{r['name']}: "
        f"{r['bicubic_psnr']:.2f} → "
        f"{r['cnn_psnr']:.2f} "
        f"({r['psnr_gain']:+.2f} dB), "
        f"SSIM {r['ssim_gain']:+.4f}, "
        f"residual RMS={r['residual_rms']:.6f}"
    )


# ============================================================
# DETAILED BEST CASE
# ============================================================

print("\n" + "=" * 70)
print("BEST CASE DETAILED AUDIT")
print("=" * 70)

print(
    "\nROI:",
    best["name"]
)

print(
    "Bicubic:",
    f"{best['bicubic_psnr']:.4f} dB"
)

print(
    "CNN:",
    f"{best['cnn_psnr']:.4f} dB"
)

print(
    "Gain:",
    f"{best['psnr_gain']:+.4f} dB"
)

print(
    "\nBicubic SSIM:",
    f"{best['bicubic_ssim']:.4f}"
)

print(
    "CNN SSIM:",
    f"{best['cnn_ssim']:.4f}"
)

print(
    "SSIM gain:",
    f"{best['ssim_gain']:+.4f}"
)

print("\nPer-band PSNR:")

band_names = [
    "B04 Red",
    "B03 Green",
    "B02 Blue",
    "B08 NIR",
]

for i, name in enumerate(band_names):

    print(
        f"{name}: "
        f"{best['band_b_psnr'][i]:.4f} → "
        f"{best['band_c_psnr'][i]:.4f} "
        f"("
        f"{best['band_c_psnr'][i] - best['band_b_psnr'][i]:+.4f}"
        f")"
    )

print("\nCNN residual:")

print(
    "Mean absolute:",
    f"{best['residual_mae']:.8f}"
)

print(
    "RMS:",
    f"{best['residual_rms']:.8f}"
)

print(
    "95th percentile:",
    f"{best['residual_p95']:.8f}"
)


# ============================================================
# METADATA FOR BEST CASE
# ============================================================

best_metadata = source_info.get(
    best["name"]
)

print("\nBest-case metadata:")

if best_metadata:

    print(
        json.dumps(
            best_metadata,
            indent=2,
        )
    )

else:

    print(
        "Metadata not found for this ROI."
    )


# ============================================================
# SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECK")
print("=" * 70)

if best["name"] not in roi_to_split:

    print(
        "WARNING: Best ROI is not present in processed split map."
    )

else:

    print(
        "Best ROI split:",
        roi_to_split[best["name"]]
    )

if roi_to_split.get(
    best["name"]
) != "test":

    print(
        "WARNING: Best ROI is NOT in test split!"
    )

else:

    print(
        "✅ Best ROI is in test split."
    )


print("\n✅ Audit complete.")