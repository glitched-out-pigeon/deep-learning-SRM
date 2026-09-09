from pathlib import Path
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "highres"
    / "bengaluru_20260427"
)

BANDS = [
    "B02",
    "B03",
    "B04",
    "B08_aligned",
]


print("=" * 70)
print("BENGALURU BAND ALIGNMENT CHECK")
print("=" * 70)


reference = None


for band in BANDS:

    path = SCENE_DIR / f"{band}.tiff"

    with rasterio.open(path) as src:

        print("\n" + "-" * 70)
        print(band)
        print("-" * 70)

        print("Shape:")
        print(
            f"  {src.width} × {src.height}"
        )

        print("CRS:")
        print(
            f"  {src.crs}"
        )

        print("Transform:")
        print(
            f"  {src.transform}"
        )

        print("Resolution:")
        print(
            f"  {src.res}"
        )

        print("Bounds:")
        print(
            f"  {src.bounds}"
        )

        current = {
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "res": src.res,
        }

        if reference is None:

            reference = current

            print("\nUsed as reference.")

        else:

            print("\nDifferences from B02:")

            print(
                "  Dimensions:",
                current["width"] == reference["width"]
                and current["height"] == reference["height"]
            )

            print(
                "  CRS:",
                current["crs"] == reference["crs"]
            )

            print(
                "  Resolution:",
                current["res"] == reference["res"]
            )

            print(
                "  Transform:",
                current["transform"] == reference["transform"]
            )

            print(
                "  Bounds:",
                current["bounds"] == reference["bounds"]
            )


print("\n" + "=" * 70)
print("ALIGNMENT CHECK COMPLETE")
print("=" * 70)