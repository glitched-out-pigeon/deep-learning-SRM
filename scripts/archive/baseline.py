from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import SuperResolutionDataset


DATA_ROOT = PROJECT_ROOT / "data" / "processed"

BATCH_SIZE = 1

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BICUBIC BASELINE — MULTI-SCENE DATASET")
print("=" * 70)

print("\nDevice:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Test dataset
# ------------------------------------------------------------

test_dataset = SuperResolutionDataset(
    root_dir=DATA_ROOT,
    split="test",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

psnr_values = []
ssim_values = []

print("\nTest scene is the held-out TEST split.")
print(f"Test pairs: {len(test_dataset)}")

for index, (lr, hr) in enumerate(test_loader):

    lr = lr.to(device)
    hr = hr.to(device)

    # Bicubic ×4
    bicubic = F.interpolate(
        lr,
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    )

    bicubic = torch.clamp(
        bicubic,
        0.0,
        1.0,
    )

    # Convert to HWC
    hr_img = (
        hr[0]
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    bicubic_img = (
        bicubic[0]
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    psnr = peak_signal_noise_ratio(
        hr_img,
        bicubic_img,
        data_range=1.0,
    )

    ssim = structural_similarity(
        hr_img,
        bicubic_img,
        channel_axis=2,
        data_range=1.0,
    )

    psnr_values.append(psnr)
    ssim_values.append(ssim)

    filename = test_dataset.lr_files[index].name

    print(
        f"{filename} | "
        f"PSNR: {psnr:.4f} dB | "
        f"SSIM: {ssim:.4f}"
    )


# ------------------------------------------------------------
# Average
# ------------------------------------------------------------

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print("\n" + "=" * 70)
print("BICUBIC BASELINE RESULTS")
print("=" * 70)

print(f"\nAverage PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")

print("\n✅ Bicubic baseline complete.")