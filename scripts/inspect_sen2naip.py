from pathlib import Path
import zipfile
import json
import rasterio
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "sen2naip"

zip_files = list(DATA_ROOT.rglob("*.zip"))

if not zip_files:
    raise FileNotFoundError("SEN2NAIP ZIP not found.")

ZIP_PATH = zip_files[0]

print("=" * 70)
print("SEN2NAIP REAL-DATA STRUCTURE TEST")
print("=" * 70)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    # ------------------------------------------------------------
    # 1. Find ROI directories
    # ------------------------------------------------------------
    roi_names = sorted({
        Path(name).parent.name
        for name in z.namelist()
        if name.endswith("/lr.tif")
    })

    print(f"\nROIs found: {len(roi_names)}")

    # Inspect first five
    test_rois = roi_names[:5]

    for roi in test_rois:

        print("\n" + "-" * 70)
        print(f"ROI: {roi}")
        print("-" * 70)

        lr_member = f"cross-sensor/{roi}/lr.tif"
        hr_member = f"cross-sensor/{roi}/hr.tif"
        metadata_member = f"cross-sensor/{roi}/metadata.json"

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------
        if metadata_member in z.namelist():
            metadata = json.loads(
                z.read(metadata_member).decode("utf-8")
            )

            print("\nMetadata:")
            print(json.dumps(metadata, indent=2))

        # --------------------------------------------------------
        # Read TIFFs through temporary extraction
        # --------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmp:

            lr_path = Path(tmp) / "lr.tif"
            hr_path = Path(tmp) / "hr.tif"

            with z.open(lr_member) as src, open(lr_path, "wb") as dst:
                dst.write(src.read())

            with z.open(hr_member) as src, open(hr_path, "wb") as dst:
                dst.write(src.read())

            with rasterio.open(lr_path) as src:

                print("\nLR:")
                print("  Shape:", (src.count, src.height, src.width))
                print("  Dtype:", src.dtypes)
                print("  CRS:", src.crs)
                print("  Resolution:", src.res)
                print("  Bounds:", src.bounds)

            with rasterio.open(hr_path) as src:

                print("\nHR:")
                print("  Shape:", (src.count, src.height, src.width))
                print("  Dtype:", src.dtypes)
                print("  CRS:", src.crs)
                print("  Resolution:", src.res)
                print("  Bounds:", src.bounds)