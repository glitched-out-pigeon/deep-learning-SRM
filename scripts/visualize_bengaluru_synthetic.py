import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

LR_FILE = (
    DATA_ROOT
    / "train"
    / "LR"
    / "bengaluru_20260427_0000.npy"
)

HR_FILE = (
    DATA_ROOT
    / "train"
    / "HR"
    / "bengaluru_20260427_0000.npy"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "pixelshuffle_cnn_best.pth"
)

OUTPUT_FILE = (
    DATA_ROOT
    / "bengaluru_64x256_old_model_test.png"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LEGACY RESIDUAL BLOCK
# ============================================================

class LegacyResidualBlock(nn.Module):

    def __init__(self, features):

        super().__init__()

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.relu(out)

        out = self.conv2(out)

        return out + residual


# ============================================================
# LEGACY PIXELSHUFFLE UPSAMPLER
# ============================================================

class LegacyUpsample2x(nn.Module):

    def __init__(self, features):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1,
            ),

            nn.PixelShuffle(2),

        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# LEGACY MODEL
# ============================================================

class LegacyResidualPixelShuffleSR(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        features=64,
        num_blocks=8,
    ):

        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1,
        )

        self.residual_blocks = nn.ModuleList(
            [
                LegacyResidualBlock(
                    features
                )
                for _ in range(num_blocks)
            ]
        )

        self.middle_conv = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

        self.upscale_2x_1 = (
            LegacyUpsample2x(
                features
            )
        )

        self.upscale_2x_2 = (
            LegacyUpsample2x(
                features
            )
        )

        self.output_conv = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Bicubic base
        # ----------------------------------------------------

        base = F.interpolate(
            x,
            scale_factor=4,
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        feat = self.input_conv(x)

        residual = feat

        # ----------------------------------------------------
        # Residual blocks
        # ----------------------------------------------------

        for block in self.residual_blocks:

            feat = block(feat)

        # ----------------------------------------------------
        # Middle feature processing
        # ----------------------------------------------------

        feat = self.middle_conv(
            feat
        )

        feat = feat + residual

        # ----------------------------------------------------
        # 4× PixelShuffle
        # ----------------------------------------------------

        feat = self.upscale_2x_1(
            feat
        )

        feat = self.upscale_2x_2(
            feat
        )

        # ----------------------------------------------------
        # Predicted residual
        # ----------------------------------------------------

        residual_hr = self.output_conv(
            feat
        )

        # ----------------------------------------------------
        # Residual learning
        # ----------------------------------------------------

        return base + residual_hr


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    LR_FILE,
    HR_FILE,
    CHECKPOINT,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

lr = np.load(
    LR_FILE
).astype(np.float32)

hr = np.load(
    HR_FILE
).astype(np.float32)


print("=" * 70)
print("BENGALURU OLD-SYNTHETIC MODEL TEST")
print("=" * 70)

print("\nDevice:")
print(device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("\nLR:")
print(
    "Shape:",
    lr.shape
)

print(
    "Range:",
    f"{lr.min():.6f} -> "
    f"{lr.max():.6f}"
)

print("\nHR:")
print(
    "Shape:",
    hr.shape
)

print(
    "Range:",
    f"{hr.min():.6f} -> "
    f"{hr.max():.6f}"
)


# ============================================================
# VALIDATE
# ============================================================

if lr.shape != (
    64,
    64,
    3,
):

    raise RuntimeError(
        f"Expected LR (64,64,3), "
        f"got {lr.shape}"
    )

if hr.shape != (
    256,
    256,
    3,
):

    raise RuntimeError(
        f"Expected HR (256,256,3), "
        f"got {hr.shape}"
    )


# ============================================================
# LOAD LEGACY MODEL
# ============================================================

model = LegacyResidualPixelShuffleSR(
    in_channels=3,
    out_channels=3,
    features=64,
    num_blocks=8,
).to(device)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)


print(
    "\nCheckpoint:"
)

print(
    CHECKPOINT
)

print(
    "Checkpoint type:",
    type(checkpoint)
)


# ============================================================
# LOAD STATE DICT
# ============================================================

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True,
)


model.eval()


# ============================================================
# PRINT SUCCESS
# ============================================================

print(
    "\n✅ Legacy checkpoint loaded successfully."
)


# ============================================================
# INFERENCE
# ============================================================

x = torch.from_numpy(
    lr.transpose(2, 0, 1)
).unsqueeze(0).to(
    device
)


with torch.no_grad():

    sr = model(x)


sr = (
    sr
    .squeeze(0)
    .cpu()
    .numpy()
    .transpose(1, 2, 0)
)

sr = np.clip(
    sr,
    0.0,
    1.0,
)


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# RGB DISPLAY
# ============================================================

def display_rgb(
    image
):

    rgb = image[:, :, :3].astype(
        np.float32
    )

    output = np.zeros_like(
        rgb
    )

    for c in range(3):

        channel = rgb[:, :, c]

        low = np.percentile(
            channel,
            2,
        )

        high = np.percentile(
            channel,
            98,
        )

        if high <= low:

            output[:, :, c] = np.clip(
                channel,
                0,
                1,
            )

        else:

            output[:, :, c] = np.clip(
                (
                    channel - low
                )
                / (
                    high - low
                ),
                0,
                1,
            )

    return output


lr_rgb = display_rgb(
    lr
)

sr_rgb = display_rgb(
    sr
)

hr_rgb = display_rgb(
    hr
)


# ============================================================
# CREATE COMPARISON
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6),
)


# ------------------------------------------------------------
# LR
# ------------------------------------------------------------

axes[0].imshow(
    lr_rgb,
    interpolation="nearest",
)

axes[0].set_title(
    "LR — 64 × 64\n"
    "Bengaluru",
    fontsize=17,
    fontweight="bold",
)


# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

axes[1].imshow(
    sr_rgb,
    interpolation="nearest",
)

axes[1].set_title(
    "CNN SR — 256 × 256\n"
    f"{psnr:.2f} dB PSNR | "
    f"{ssim:.3f} SSIM",
    fontsize=17,
    fontweight="bold",
)


# ------------------------------------------------------------
# HR
# ------------------------------------------------------------

axes[2].imshow(
    hr_rgb,
    interpolation="nearest",
)

axes[2].set_title(
    "HR Target — 256 × 256\n"
    "Bengaluru",
    fontsize=17,
    fontweight="bold",
)


# ============================================================
# CLEAN FIGURE
# ============================================================

for ax in axes:

    ax.axis("off")


fig.suptitle(
    "Deep Learning Based Super-Resolution Mapping\n"
    "Bengaluru — 64×64 → 256×256",
    fontsize=20,
    fontweight="bold",
)


plt.tight_layout(
    rect=(0, 0, 1, 0.91)
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    "\nCNN output:",
    sr.shape
)

print(
    "PSNR:",
    f"{psnr:.4f} dB"
)

print(
    "SSIM:",
    f"{ssim:.4f}"
)

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    "\n✅ Exact old Bengaluru LR → old CNN → old HR comparison created."
)