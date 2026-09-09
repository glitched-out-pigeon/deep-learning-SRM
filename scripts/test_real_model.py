import sys
from pathlib import Path

import torch


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.sr_cnn import ResidualPixelShuffleSR


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# ============================================================

model = ResidualPixelShuffleSR(
    in_channels=4,
    out_channels=4,
    features=64,
    num_blocks=8,
).to(device)


# ============================================================
# TEST INPUT
# ============================================================

x = torch.randn(
    2,
    4,
    121,
    121,
    device=device,
)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():
    y = model(x)


# ============================================================
# PARAMETER COUNT
# ============================================================

parameters = sum(
    p.numel()
    for p in model.parameters()
)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 70)
print("REAL SEN2NAIP MODEL TEST")
print("=" * 70)

print("\nDevice:")
print(device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("\nInput:")
print(x.shape)

print("\nOutput:")
print(y.shape)

print("\nParameter count:")
print(f"{parameters:,}")

print("\nExpected:")
print("(2, 4, 121, 121) -> (2, 4, 484, 484)")

print("\nOutput range:")
print(
    f"{y.min().item():.6f} -> "
    f"{y.max().item():.6f}"
)

if y.shape != (2, 4, 484, 484):
    raise RuntimeError(
        f"Unexpected output shape: {y.shape}"
    )

print("\n✅ 4-channel PixelShuffle model works.")