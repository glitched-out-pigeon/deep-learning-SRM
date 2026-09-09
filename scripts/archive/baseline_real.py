from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LR_DIR = PROJECT_ROOT / "data" / "processed" / "real" / "test" / "LR"
HR_DIR = PROJECT_ROOT / "data" / "processed" / "real" / "test" / "HR"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("REAL SEN2NAIP BICUBIC BASELINE")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# FILES
# ============================================================

lr_files = sorted(LR_DIR.glob("*.npy"))
hr_files = sorted(HR_DIR.glob("*.npy"))

if not lr_files:
    raise RuntimeError("No LR test files found.")

if not hr_files:
    raise RuntimeError("No HR test files found.")

if len(lr_files) != len(hr_files):
    raise RuntimeError(
        f"LR/HR count mismatch: "
        f"{len(lr_files)} vs {len(hr_files)}"
    )

print("\nTest pairs:", len(lr_files))


# ============================================================
# EVALUATION
# ============================================================

psnr_values = []
ssim_values = []


for index, lr_path in enumerate(lr_files):

    hr_path = HR_DIR / lr_path.name

    if not hr_path.exists():
        raise FileNotFoundError(
            f"Missing HR pair for {lr_path.name}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    lr = np.load(lr_path).astype(np.float32)
    hr = np.load(hr_path).astype(np.float32)

    # HWC → NCHW
    lr_tensor = torch.from_numpy(
        lr.transpose(2, 0, 1)
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Bicubic 4x
    # --------------------------------------------------------

    lr_tensor = lr_tensor.to(DEVICE)

    with torch.no_grad():

        sr = F.interpolate(
            lr_tensor,
            size=(hr.shape[0], hr.shape[1]),
            mode="bicubic",
            align_corners=False,
        )

    sr = sr.squeeze(0).cpu().numpy()

    # CHW → HWC
    sr = np.transpose(sr, (1, 2, 0))

    sr = np.clip(sr, 0.0, 1.0)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    psnr = peak_signal_noise_ratio(
        hr,
        sr,
        data_range=1.0,
    )

    ssim = structural_similarity(
        hr,
        sr,
        channel_axis=2,
        data_range=1.0,
    )

    psnr_values.append(psnr)
    ssim_values.append(ssim)

    if index < 10 or (index + 1) % 50 == 0:

        print(
            f"{index + 1:3d}/{len(lr_files)} "
            f"{lr_path.stem}: "
            f"PSNR={psnr:.4f} dB "
            f"SSIM={ssim:.4f}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL REAL-DATA BICUBIC RESULTS")
print("=" * 70)

print(
    f"\nAverage PSNR: "
    f"{np.mean(psnr_values):.4f} dB"
)

print(
    f"Average SSIM: "
    f"{np.mean(ssim_values):.4f}"
)

print(
    f"\nBest PSNR: "
    f"{np.max(psnr_values):.4f} dB"
)

print(
    f"Worst PSNR: "
    f"{np.min(psnr_values):.4f} dB"
)

print(
    f"\nBest SSIM: "
    f"{np.max(ssim_values):.4f}"
)

print(
    f"Worst SSIM: "
    f"{np.min(ssim_values):.4f}"
)

print("\n✅ Real bicubic baseline complete.")