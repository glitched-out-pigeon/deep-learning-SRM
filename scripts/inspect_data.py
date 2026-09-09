from pathlib import Path

import numpy as np
import rasterio


DATA_DIR = Path("data/raw/highres/bengaluru_20260427")

bands = ["B02", "B03", "B04"]


def inspect_band(name: str) -> None:
    path = DATA_DIR / f"{name}.tiff"

    print("\n" + "=" * 60)
    print(f"{name} -> {path}")
    print("=" * 60)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    with rasterio.open(path) as src:
        data = src.read(1, masked=True)

        print(f"Width:          {src.width}")
        print(f"Height:         {src.height}")
        print(f"Band count:     {src.count}")
        print(f"Data type:      {src.dtypes[0]}")
        print(f"CRS:             {src.crs}")
        print(f"Resolution:     {src.res}")
        print(f"NoData:          {src.nodata}")
        print(f"Bounds:          {src.bounds}")

        values = data.compressed()

        if values.size == 0:
            print("WARNING: No valid pixels found.")
            return

        print(f"Valid pixels:   {values.size}")
        print(f"Min:             {values.min():.6f}")
        print(f"Max:             {values.max():.6f}")
        print(f"Mean:            {values.mean():.6f}")
        print(f"Median:          {np.median(values):.6f}")
        print(f"Std:             {values.std():.6f}")


def main() -> None:
    print("Sentinel-2 dataset inspection")
    print(f"Data directory: {DATA_DIR.resolve()}")

    for band in bands:
        inspect_band(band)

    print("\n" + "=" * 60)
    print("CHECKING BAND ALIGNMENT")
    print("=" * 60)

    metadata = {}

    for band in bands:
        path = DATA_DIR / f"{band}.tiff"

        with rasterio.open(path) as src:
            metadata[band] = {
                "width": src.width,
                "height": src.height,
                "crs": src.crs,
                "transform": src.transform,
                "resolution": src.res,
            }

    reference = metadata["B02"]

    all_match = True

    for band in bands[1:]:
        current = metadata[band]

        checks = {
            "dimensions": (
                current["width"] == reference["width"]
                and current["height"] == reference["height"]
            ),
            "CRS": current["crs"] == reference["crs"],
            "transform": current["transform"] == reference["transform"],
            "resolution": current["resolution"] == reference["resolution"],
        }

        print(f"\n{band} vs B02:")
        for check, result in checks.items():
            print(f"  {check:<12}: {'OK' if result else 'MISMATCH'}")

        if not all(checks.values()):
            all_match = False

    print("\n" + "=" * 60)
    print(
        "RESULT:",
        "ALL BANDS ALIGN CORRECTLY ✅"
        if all_match
        else "BAND ALIGNMENT PROBLEM FOUND ❌",
    )
    print("=" * 60)


if __name__ == "__main__":
    main()