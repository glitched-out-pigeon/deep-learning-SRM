from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root so Python can find models/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.sr_cnn import SRCNN4x


# ============================================================
# Paths
# ============================================================

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "residual_cnn_gradient_best.pth"

TEST_LR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "LR"
)

TEST_HR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "HR"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gradient_cnn_result.png"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SUPER-RESOLUTION INFERENCE")
print("=" * 60)

print("\nDevice:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Check files
# ============================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

lr_files = sorted(TEST_LR_DIR.glob("*.npy"))

if not lr_files:
    raise RuntimeError(
        f"No LR test files found in:\n{TEST_LR_DIR}"
    )


# Use the first test sample
lr_path = lr_files[0]
hr_path = TEST_HR_DIR / lr_path.name

if not hr_path.exists():
    raise FileNotFoundError(
        f"Matching HR file not found:\n{hr_path}"
    )

print("\nTest sample:")
print("LR:", lr_path.name)
print("HR:", hr_path.name)


# ============================================================
# Load model
# ============================================================

model = SRCNN4x().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "\nLoaded checkpoint from epoch:",
    checkpoint["epoch"],
)

print(
    "Checkpoint validation loss:",
    f"{checkpoint['val_loss']:.6f}",
)


# ============================================================
# Load images
# ============================================================

lr = np.load(lr_path).astype(np.float32)
hr = np.load(hr_path).astype(np.float32)

print("\nOriginal arrays:")
print("LR:", lr.shape)
print("HR:", hr.shape)


# ============================================================
# Convert HWC -> CHW -> Batch
# ============================================================

lr_tensor = torch.from_numpy(
    np.transpose(lr, (2, 0, 1))
).unsqueeze(0)

lr_tensor = lr_tensor.to(device)


# ============================================================
# Run inference
# ============================================================

with torch.no_grad():
    sr_tensor = model(lr_tensor)


# ============================================================
# Convert SR tensor back to image
# ============================================================

sr = sr_tensor.squeeze(0).cpu().numpy()

sr = np.transpose(sr, (1, 2, 0))

sr = np.clip(sr, 0.0, 1.0)

print("\nSuper-resolved output:")
print("SR:", sr.shape)
print(
    "SR range:",
    f"{sr.min():.6f} -> {sr.max():.6f}",
)


# ============================================================
# Create comparison
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)

axes[0].imshow(np.clip(lr, 0, 1))
axes[0].set_title("LR — 64 × 64")
axes[0].axis("off")

axes[1].imshow(sr)
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

print("\nComparison saved to:")
print(OUTPUT_PATH)

print("\n✅ Inference complete!")