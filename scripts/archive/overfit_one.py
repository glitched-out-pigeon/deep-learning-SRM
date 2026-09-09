from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.sr_cnn import SRCNN4x


# ============================================================
# Configuration
# ============================================================

LR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "LR"
    / "bengaluru_20260427_0000.npy"
)

HR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "HR"
    / "bengaluru_20260427_0000.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "overfit_result.png"
)

EPOCHS = 200
LEARNING_RATE = 1e-4


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ONE-IMAGE OVERFIT TEST")
print("=" * 60)

print("\nDevice:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load data
# ============================================================

lr = np.load(LR_PATH).astype(np.float32)
hr = np.load(HR_PATH).astype(np.float32)

print("\nLoaded:")
print("LR:", lr.shape)
print("HR:", hr.shape)


# HWC -> CHW -> Batch
lr_tensor = torch.from_numpy(
    np.transpose(lr, (2, 0, 1))
).unsqueeze(0).to(device)

hr_tensor = torch.from_numpy(
    np.transpose(hr, (2, 0, 1))
).unsqueeze(0).to(device)

print("LR tensor:", lr_tensor.shape)
print("HR tensor:", hr_tensor.shape)


# ============================================================
# Model
# ============================================================

model = SRCNN4x().to(device)

criterion = nn.L1Loss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training
# ============================================================

print("\nStarting overfit test...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    sr = model(lr_tensor)

    loss = criterion(sr, hr_tensor)

    loss.backward()

    optimizer.step()

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch [{epoch:3d}/{EPOCHS}] "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# Final inference
# ============================================================

model.eval()

with torch.no_grad():
    sr_tensor = model(lr_tensor)

sr = (
    sr_tensor
    .squeeze(0)
    .cpu()
    .numpy()
)

sr = np.transpose(sr, (1, 2, 0))

sr_display = np.clip(sr, 0.0, 1.0)

final_loss = criterion(
    sr_tensor,
    hr_tensor,
).item()

print("\nFinal loss:", f"{final_loss:.6f}")
print(
    "SR range:",
    f"{sr.min():.6f} -> {sr.max():.6f}"
)


# ============================================================
# Save visual comparison
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)

axes[0].imshow(np.clip(lr, 0, 1))
axes[0].set_title("LR — 64 × 64")
axes[0].axis("off")

axes[1].imshow(sr_display)
axes[1].set_title("CNN SR — 256 × 256")
axes[1].axis("off")

axes[2].imshow(np.clip(hr, 0, 1))
axes[2].set_title("HR Target — 256 × 256")
axes[2].axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("\nResult saved to:")
print(OUTPUT_PATH)

print("\n✅ One-image overfit test complete!")