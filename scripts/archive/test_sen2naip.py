from pathlib import Path

import numpy as np
import rioxarray


DEMO_BASE = (
    "https://huggingface.co/datasets/"
    "isp-uv-es/SEN2NAIP/resolve/main/demo/"
    "cross-sensor/ROI_0000/"
)

LR_URL = DEMO_BASE + "lr.tif"
HR_URL = DEMO_BASE + "hr.tif"


print("=" * 70)
print("SEN2NAIP REAL-DATA TEST")
print("=" * 70)

print("\nLoading LR:")
print(LR_URL)

lr = rioxarray.open_rasterio(LR_URL)

print("\nLR:")
print("Shape:", lr.shape)
print("Dtype:", lr.dtype)
print("CRS:", lr.rio.crs)
print("Resolution:", lr.rio.resolution())
print(
    "Range:",
    float(lr.min()),
    "->",
    float(lr.max()),
)

print("\nLoading HR:")
print(HR_URL)

hr = rioxarray.open_rasterio(HR_URL)

print("\nHR:")
print("Shape:", hr.shape)
print("Dtype:", hr.dtype)
print("CRS:", hr.rio.crs)
print("Resolution:", hr.rio.resolution())
print(
    "Range:",
    float(hr.min()),
    "->",
    float(hr.max()),
)

print("\n" + "=" * 70)

print("LR dimensions:", lr.sizes)
print("HR dimensions:", hr.sizes)

print("\n✅ SEN2NAIP demo loaded successfully.")