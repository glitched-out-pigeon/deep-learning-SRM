import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.dataset_real import SEN2NAIPDataset


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

dataset = SEN2NAIPDataset(
    root=DATA_ROOT,
    split="train",
    lr_crop_size=64,
    training=True,
)


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# FIRST BATCH
# ============================================================

batch = next(iter(loader))

lr = batch["lr"]
hr = batch["hr"]

print("=" * 70)
print("SEN2NAIP REAL DATALOADER TEST")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("\nDataset size:")
print(len(dataset))

print("\nBatch:")
print("LR:", lr.shape)
print("HR:", hr.shape)

print("\nDtypes:")
print("LR:", lr.dtype)
print("HR:", hr.dtype)

print("\nRanges:")
print(
    "LR:",
    f"{lr.min().item():.6f} -> "
    f"{lr.max().item():.6f}"
)

print(
    "HR:",
    f"{hr.min().item():.6f} -> "
    f"{hr.max().item():.6f}"
)

# ------------------------------------------------------------
# GPU test
# ------------------------------------------------------------

lr_gpu = lr.to(device, non_blocking=True)
hr_gpu = hr.to(device, non_blocking=True)

print("\nGPU tensors:")
print("LR:", lr_gpu.shape, lr_gpu.device)
print("HR:", hr_gpu.shape, hr_gpu.device)

# ------------------------------------------------------------
# Shape assertions
# ------------------------------------------------------------

assert lr.shape == (8, 4, 64, 64)
assert hr.shape == (8, 4, 256, 256)

assert lr.dtype == torch.float32
assert hr.dtype == torch.float32

assert 0.0 <= lr.min() <= 1.0
assert 0.0 <= lr.max() <= 1.0

assert 0.0 <= hr.min() <= 1.0
assert 0.0 <= hr.max() <= 1.0

assert lr_gpu.device.type == device.type
assert hr_gpu.device.type == device.type

print("\n✅ Real SEN2NAIP DataLoader test passed.")