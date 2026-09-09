from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

hr_path = ROOT / "data" / "processed" / "train" / "HR" / "bengaluru_20260427_0000.npy"
lr_path = ROOT / "data" / "processed" / "train" / "LR" / "bengaluru_20260427_0000.npy"

hr = np.load(hr_path)
lr = np.load(lr_path)

print("HR shape:", hr.shape)
print("HR dtype:", hr.dtype)
print("HR range:", hr.min(), "to", hr.max())

print("LR shape:", lr.shape)
print("LR dtype:", lr.dtype)
print("LR range:", lr.min(), "to", lr.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(np.clip(hr, 0, 1))
axes[0].set_title("HR — 256 × 256")
axes[0].axis("off")

axes[1].imshow(np.clip(lr, 0, 1))
axes[1].set_title("LR — 64 × 64")
axes[1].axis("off")

plt.tight_layout()

output = ROOT / "data" / "processed" / "dataset_preview.png"
plt.savefig(output, dpi=150, bbox_inches="tight")

print("\nPreview saved to:")
print(output)