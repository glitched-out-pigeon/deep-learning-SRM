# ============================================================
# BENGALURU SATELLITE SUPER-RESOLUTION
# Improved 3-Channel SR Demo
#
# Input  : 64 x 64 x 3
# Output : 256 x 256 x 3
#
# Main improvements:
#   1. 96 feature channels
#   2. 12 residual blocks
#   3. ICNR initialization for PixelShuffle
#   4. L1 + SSIM + Gradient + Color loss
#   5. Controlled residual output
#   6. Realistic LR degradation
#   7. Training/inference degradation consistency
#   8. Random flip/rotation augmentation
#   9. ReduceLROnPlateau scheduler
#  10. Mixed precision for RTX 3060
#
# Final visualization:
#
#   LOW-RES INPUT | OUR CNN OUTPUT | HR TARGET
# ============================================================

import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

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

TRAIN_HR_DIR = (
    DATA_ROOT
    / "train"
    / "HR"
)

VAL_HR_DIR = (
    DATA_ROOT
    / "val"
    / "HR"
)

# Prefer an unseen test patch for the final demo.
TEST_HR_DIR = (
    DATA_ROOT
    / "test"
    / "HR"
)

OUTPUT_DIR = (
    DATA_ROOT
    / "bengaluru_demo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "bengaluru_demo_3ch_icnr.pth"
)

OUTPUT_IMAGE_PATH = (
    OUTPUT_DIR
    / "bengaluru_lr_cnn_hr_icnr.png"
)

OUTPUT_LR_PATH = (
    OUTPUT_DIR
    / "bengaluru_demo_lr.npy"
)

OUTPUT_SR_PATH = (
    OUTPUT_DIR
    / "bengaluru_demo_sr.npy"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEMO_SEED = 1234

BATCH_SIZE = 8

EPOCHS = 250

LEARNING_RATE = 2e-4

MIN_LEARNING_RATE = 1e-6

FEATURES = 96

NUM_BLOCKS = 12

SCALE = 4

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    SEED
)

np.random.seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ============================================================
# DATASET
# ============================================================

class SyntheticSatelliteDataset(
    Dataset
):
    """
    Loads 256x256 HR satellite patches.

    LR images are generated from HR on the fly.

    Training degradation:
        HR
        ↓
        Gaussian blur
        ↓
        bicubic downsampling
        ↓
        sensor noise
        ↓
        quantization
        ↓
        LR 64x64

    This means the model sees different degraded
    versions throughout training.
    """

    def __init__(
        self,
        hr_dir,
        training=False,
        deterministic=False,
    ):

        self.hr_dir = Path(
            hr_dir
        )

        self.training = training

        self.deterministic = (
            deterministic
        )

        self.files = sorted(
            self.hr_dir.glob(
                "*.npy"
            )
        )

        if len(self.files) == 0:

            raise RuntimeError(
                "\nNo HR .npy files found in:\n"
                f"{self.hr_dir}\n"
            )

    def __len__(self):

        return len(
            self.files
        )

    # ========================================================
    # AUGMENTATION
    # ========================================================

    def augment(
        self,
        hr,
    ):

        # Horizontal flip
        if np.random.rand() < 0.5:

            hr = np.flip(
                hr,
                axis=1,
            ).copy()

        # Vertical flip
        if np.random.rand() < 0.5:

            hr = np.flip(
                hr,
                axis=0,
            ).copy()

        # Random 90-degree rotation
        k = np.random.randint(
            0,
            4,
        )

        if k != 0:

            hr = np.rot90(
                hr,
                k=k,
                axes=(0, 1),
            ).copy()

        return hr

    # ========================================================
    # GAUSSIAN BLUR
    # ========================================================

    def gaussian_blur(
        self,
        image,
        sigma,
    ):
        """
        Applies a depthwise Gaussian blur.
        """

        kernel_size = 5

        radius = (
            kernel_size - 1
        ) / 2

        coords = torch.arange(
            -radius,
            radius + 1,
            dtype=torch.float32,
        )

        gaussian = torch.exp(
            -(
                coords ** 2
            )
            / (
                2.0
                * sigma
                * sigma
            )
        )

        gaussian = (
            gaussian
            / gaussian.sum()
        )

        kernel_2d = (
            gaussian[:, None]
            * gaussian[None, :]
        )

        kernel = (
            kernel_2d
            .expand(
                3,
                1,
                kernel_size,
                kernel_size,
            )
        )

        kernel = kernel.to(
            device=image.device,
            dtype=image.dtype,
        )

        return F.conv2d(
            image,
            kernel,
            padding=2,
            groups=3,
        )

    # ========================================================
    # DEGRADATION
    # ========================================================

    def degrade(
        self,
        hr,
        randomize=True,
    ):
        """
        Creates a 64x64 LR observation.

        When randomize=True:
            random blur/noise/quantization are used.

        When randomize=False:
            fixed parameters are used for deterministic
            validation/demo behavior.
        """

        hr_tensor = torch.from_numpy(
            hr.transpose(2, 0, 1)
        ).float()

        hr_tensor = (
            hr_tensor
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # BLUR
        # ----------------------------------------------------

        if randomize:

            apply_blur = (
                np.random.rand()
                < 0.90
            )

            if apply_blur:

                sigma = np.random.uniform(
                    0.7,
                    1.6,
                )

                hr_tensor = (
                    self.gaussian_blur(
                        hr_tensor,
                        sigma,
                    )
                )

        else:

            # Fixed blur used by validation/demo
            sigma = 1.0

            hr_tensor = (
                self.gaussian_blur(
                    hr_tensor,
                    sigma,
                )
            )

        # ----------------------------------------------------
        # 4x DOWNSAMPLE
        # ----------------------------------------------------

        lr_tensor = F.interpolate(
            hr_tensor,
            size=(64, 64),
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # SENSOR NOISE
        # ----------------------------------------------------

        if randomize:

            noise_std = (
                np.random.uniform(
                    0.0,
                    0.015,
                )
            )

        else:

            # Fixed validation/demo noise
            noise_std = 0.0075

        if noise_std > 0:

            noise = (
                torch.randn_like(
                    lr_tensor
                )
                * noise_std
            )

            lr_tensor = (
                lr_tensor
                + noise
            )

        # ----------------------------------------------------
        # QUANTIZATION
        # ----------------------------------------------------

        if randomize:

            quantize = (
                np.random.rand()
                < 0.70
            )

        else:

            quantize = True

        if quantize:

            lr_tensor = (
                torch.round(
                    lr_tensor
                    * 255.0
                )
                / 255.0
            )

        # ----------------------------------------------------
        # CLAMP
        # ----------------------------------------------------

        lr_tensor = torch.clamp(
            lr_tensor,
            0.0,
            1.0,
        )

        return (
            lr_tensor
            .squeeze(0)
        )

    # ========================================================
    # ITEM
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        hr_file = (
            self.files[index]
        )

        hr = np.load(
            hr_file
        ).astype(
            np.float32
        )

        if hr.shape != (
            256,
            256,
            3,
        ):

            raise RuntimeError(
                "Unexpected HR shape "
                f"{hr.shape} in "
                f"{hr_file.name}"
            )

        hr = np.clip(
            hr,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # Training augmentation
        # ----------------------------------------------------

        if self.training:

            hr = self.augment(
                hr
            )

        # ----------------------------------------------------
        # LR generation
        # ----------------------------------------------------

        lr = self.degrade(
            hr,
            randomize=(
                not self.deterministic
            ),
        )

        # ----------------------------------------------------
        # HR tensor
        # ----------------------------------------------------

        hr_tensor = torch.from_numpy(
            hr.transpose(2, 0, 1)
        ).float()

        return (
            lr,
            hr_tensor,
            hr_file.name,
        )


# ============================================================
# ICNR INITIALIZATION
# ============================================================

def icnr_init(
    conv,
    scale=2,
):
    """
    ICNR initialization for sub-pixel convolution.

    This makes PixelShuffle start from a smooth,
    interpolation-like behavior rather than producing
    strong checkerboard artifacts from random sub-pixel
    filters.
    """

    out_channels, in_channels, h, w = (
        conv.weight.shape
    )

    required_channels = (
        scale ** 2
    )

    if (
        out_channels
        % required_channels
        != 0
    ):

        raise ValueError(
            "Output channels must be "
            "divisible by scale^2 for ICNR."
        )

    sub_channels = (
        out_channels
        // required_channels
    )

    # Create the base kernel
    sub_kernel = torch.empty(
        sub_channels,
        in_channels,
        h,
        w,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )

    # Kaiming initialization
    nn.init.kaiming_normal_(
        sub_kernel,
        mode="fan_out",
        nonlinearity="relu",
    )

    # Repeat each base filter scale^2 times
    sub_kernel = (
        sub_kernel
        .repeat_interleave(
            required_channels,
            dim=0,
        )
    )

    with torch.no_grad():

        conv.weight.copy_(
            sub_kernel
        )

        if conv.bias is not None:

            conv.bias.zero_()


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(
    nn.Module
):

    def __init__(
        self,
        features,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x,
    ):

        residual = x

        out = self.conv1(
            x
        )

        out = self.relu(
            out
        )

        out = self.conv2(
            out
        )

        return (
            residual
            + out
        )


# ============================================================
# ICNR PIXELSHUFFLE UPSAMPLER
# ============================================================

class Upsample2x(
    nn.Module
):

    def __init__(
        self,
        features,
    ):

        super().__init__()

        # ----------------------------------------------------
        # PixelShuffle input convolution
        # ----------------------------------------------------

        conv = nn.Conv2d(
            features,
            features * 4,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # ICNR initialization
        # ----------------------------------------------------

        icnr_init(
            conv,
            scale=2,
        )

        self.block = nn.Sequential(
            conv,

            nn.PixelShuffle(
                2
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.block(
            x
        )


# ============================================================
# SUPER-RESOLUTION MODEL
# ============================================================

class BengaluruSR(
    nn.Module
):

    def __init__(
        self,
        features=96,
        num_blocks=12,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        self.input_conv = nn.Conv2d(
            3,
            features,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # Residual trunk
        # ----------------------------------------------------

        self.residual_blocks = (
            nn.ModuleList(
                [
                    ResidualBlock(
                        features
                    )
                    for _ in range(
                        num_blocks
                    )
                ]
            )
        )

        # ----------------------------------------------------
        # Trunk output
        # ----------------------------------------------------

        self.middle_conv = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        self.upscale_2x_1 = (
            Upsample2x(
                features
            )
        )

        self.upscale_2x_2 = (
            Upsample2x(
                features
            )
        )

        # ----------------------------------------------------
        # Output residual
        # ----------------------------------------------------

        self.output_conv = nn.Conv2d(
            features,
            3,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x,
    ):

        # ----------------------------------------------------
        # Bicubic base
        # ----------------------------------------------------

        base = F.interpolate(
            x,
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        feat = self.input_conv(
            x
        )

        skip = feat

        # ----------------------------------------------------
        # Residual blocks
        # ----------------------------------------------------

        for block in (
            self.residual_blocks
        ):

            feat = block(
                feat
            )

        # ----------------------------------------------------
        # Global residual
        # ----------------------------------------------------

        feat = self.middle_conv(
            feat
        )

        feat = (
            feat
            + skip
        )

        # ----------------------------------------------------
        # 2x
        # ----------------------------------------------------

        feat = (
            self.upscale_2x_1(
                feat
            )
        )

        # ----------------------------------------------------
        # 2x again
        # ----------------------------------------------------

        feat = (
            self.upscale_2x_2(
                feat
            )
        )

        # ----------------------------------------------------
        # Learn detail residual
        # ----------------------------------------------------

        residual = (
            self.output_conv(
                feat
            )
        )

        # ----------------------------------------------------
        # COLOR / ARTIFACT CONTROL
        #
        # The CNN is only allowed to modify the bicubic
        # image within a controlled range.
        # ----------------------------------------------------

        residual = (
            0.10
            * torch.tanh(
                residual
            )
        )

        # ----------------------------------------------------
        # Final SR
        # ----------------------------------------------------

        output = (
            base
            + residual
        )

        return output


# ============================================================
# SSIM
# ============================================================

def gaussian_window(
    window_size=11,
    sigma=1.5,
    channels=3,
):

    coords = torch.arange(
        window_size,
        dtype=torch.float32,
    )

    coords = (
        coords
        - (
            window_size - 1
        ) / 2
    )

    gaussian = torch.exp(
        -(
            coords ** 2
        )
        / (
            2.0
            * sigma
            * sigma
        )
    )

    gaussian = (
        gaussian
        / gaussian.sum()
    )

    window = (
        gaussian[:, None]
        @ gaussian[None, :]
    )

    window = (
        window
        .unsqueeze(0)
        .unsqueeze(0)
    )

    window = window.expand(
        channels,
        1,
        window_size,
        window_size,
    )

    return window


def ssim(
    img1,
    img2,
    window_size=11,
    sigma=1.5,
):

    channels = (
        img1.size(1)
    )

    window = gaussian_window(
        window_size,
        sigma,
        channels,
    )

    window = window.to(
        img1.device,
        dtype=img1.dtype,
    )

    pad = (
        window_size
        // 2
    )

    mu1 = F.conv2d(
        img1,
        window,
        padding=pad,
        groups=channels,
    )

    mu2 = F.conv2d(
        img2,
        window,
        padding=pad,
        groups=channels,
    )

    mu1_sq = (
        mu1
        * mu1
    )

    mu2_sq = (
        mu2
        * mu2
    )

    mu1_mu2 = (
        mu1
        * mu2
    )

    sigma1_sq = (
        F.conv2d(
            img1 * img1,
            window,
            padding=pad,
            groups=channels,
        )
        - mu1_sq
    )

    sigma2_sq = (
        F.conv2d(
            img2 * img2,
            window,
            padding=pad,
            groups=channels,
        )
        - mu2_sq
    )

    sigma12 = (
        F.conv2d(
            img1 * img2,
            window,
            padding=pad,
            groups=channels,
        )
        - mu1_mu2
    )

    C1 = (
        0.01 ** 2
    )

    C2 = (
        0.03 ** 2
    )

    numerator = (
        (
            2
            * mu1_mu2
            + C1
        )
        * (
            2
            * sigma12
            + C2
        )
    )

    denominator = (
        (
            mu1_sq
            + mu2_sq
            + C1
        )
        * (
            sigma1_sq
            + sigma2_sq
            + C2
        )
    )

    score = (
        numerator
        / (
            denominator
            + 1e-8
        )
    )

    return score.mean()


# ============================================================
# GRADIENT LOSS
# ============================================================

def gradient_loss(
    prediction,
    target,
):

    pred_x = (
        prediction[:, :, :, 1:]
        - prediction[:, :, :, :-1]
    )

    pred_y = (
        prediction[:, :, 1:, :]
        - prediction[:, :, :-1, :]
    )

    target_x = (
        target[:, :, :, 1:]
        - target[:, :, :, :-1]
    )

    target_y = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    loss_x = F.l1_loss(
        pred_x,
        target_x,
    )

    loss_y = F.l1_loss(
        pred_y,
        target_y,
    )

    return (
        loss_x
        + loss_y
    ) / 2.0


# ============================================================
# COLOR CONSISTENCY LOSS
# ============================================================

def color_consistency_loss(
    prediction,
    target,
):
    """
    Explicitly discourages color drift.

    Compares:
        per-channel mean
        per-channel standard deviation
    """

    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------

    pred_mean = prediction.mean(
        dim=(2, 3)
    )

    target_mean = target.mean(
        dim=(2, 3)
    )

    mean_loss = F.l1_loss(
        pred_mean,
        target_mean,
    )

    # --------------------------------------------------------
    # Standard deviations
    # --------------------------------------------------------

    pred_std = prediction.std(
        dim=(2, 3)
    )

    target_std = target.std(
        dim=(2, 3)
    )

    std_loss = F.l1_loss(
        pred_std,
        target_std,
    )

    return (
        mean_loss
        + std_loss
    )


# ============================================================
# TOTAL LOSS
# ============================================================

def compute_loss(
    prediction,
    target,
):

    # --------------------------------------------------------
    # Pixel reconstruction
    # --------------------------------------------------------

    l1 = F.l1_loss(
        prediction,
        target,
    )

    # --------------------------------------------------------
    # Structural similarity
    # --------------------------------------------------------

    ssim_score = ssim(
        prediction,
        target,
    )

    ssim_loss = (
        1.0
        - ssim_score
    )

    # --------------------------------------------------------
    # Gradient / edge loss
    # --------------------------------------------------------

    grad = gradient_loss(
        prediction,
        target,
    )

    # --------------------------------------------------------
    # Color consistency
    # --------------------------------------------------------

    color = (
        color_consistency_loss(
            prediction,
            target,
        )
    )

    # --------------------------------------------------------
    # FINAL WEIGHTING
    #
    # Reduced gradient pressure compared with the previous
    # version.
    # --------------------------------------------------------

    loss = (
        0.65 * l1
        + 0.20 * ssim_loss
        + 0.05 * grad
        + 0.10 * color
    )

    return (
        loss,
        l1,
        ssim_loss,
        grad,
        color,
    )


# ============================================================
# DATASETS
# ============================================================

train_dataset = (
    SyntheticSatelliteDataset(
        TRAIN_HR_DIR,
        training=True,
        deterministic=False,
    )
)

val_dataset = (
    SyntheticSatelliteDataset(
        VAL_HR_DIR,
        training=False,
        deterministic=True,
    )
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)


# ============================================================
# MODEL
# ============================================================

model = BengaluruSR(
    features=FEATURES,
    num_blocks=NUM_BLOCKS,
).to(
    DEVICE
)


# ============================================================
# PARAMETER COUNT
# ============================================================

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(
        0.9,
        0.999,
    ),
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        min_lr=MIN_LEARNING_RATE,
    )
)


# ============================================================
# AMP
# ============================================================

if USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss_value = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0
    total_grad = 0.0
    total_color = 0.0

    batches = 0

    with torch.no_grad():

        for (
            lr,
            hr,
            _,
        ) in val_loader:

            lr = lr.to(
                DEVICE,
                non_blocking=True,
            )

            hr = hr.to(
                DEVICE,
                non_blocking=True,
            )

            if USE_AMP:

                with torch.amp.autocast(
                    "cuda"
                ):

                    prediction = (
                        model(
                            lr
                        )
                    )

                    (
                        loss,
                        l1,
                        ssim_l,
                        grad,
                        color,
                    ) = compute_loss(
                        prediction,
                        hr,
                    )

            else:

                prediction = (
                    model(
                        lr
                    )
                )

                (
                    loss,
                    l1,
                    ssim_l,
                    grad,
                    color,
                ) = compute_loss(
                    prediction,
                    hr,
                )

            total_loss_value += (
                loss.item()
            )

            total_l1 += (
                l1.item()
            )

            total_ssim_loss += (
                ssim_l.item()
            )

            total_grad += (
                grad.item()
            )

            total_color += (
                color.item()
            )

            batches += 1

    return (
        total_loss_value / batches,
        total_l1 / batches,
        total_ssim_loss / batches,
        total_grad / batches,
        total_color / batches,
    )


# ============================================================
# TRAINING HEADER
# ============================================================

print()

print("=" * 78)

print(
    "BENGALURU SATELLITE SUPER-RESOLUTION"
)

print(
    "ICNR + COLOR-STABLE 3-CHANNEL MODEL"
)

print("=" * 78)

print()

print(
    "Device:",
    DEVICE,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print()

print(
    "Training HR patches:",
    len(train_dataset),
)

print(
    "Validation HR patches:",
    len(val_dataset),
)

print(
    "Parameters:",
    f"{parameter_count:,}",
)

print(
    "Input:",
    "3 x 64 x 64",
)

print(
    "Output:",
    "3 x 256 x 256",
)

print(
    "Features:",
    FEATURES,
)

print(
    "Residual blocks:",
    NUM_BLOCKS,
)

print(
    "Epochs:",
    EPOCHS,
)

print(
    "Learning rate:",
    LEARNING_RATE,
)

print(
    "AMP:",
    USE_AMP,
)

print()

print(
    "Loss:"
)

print(
    "  0.65 L1"
)

print(
    "  0.20 SSIM"
)

print(
    "  0.05 Gradient"
)

print(
    "  0.10 Color"
)

print()

print(
    "PixelShuffle initialization: ICNR"
)

print(
    "Residual clamp: 0.10"
)

print()

print("=" * 78)


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss = float(
    "inf"
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_grad = 0.0
    running_color = 0.0

    batches = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for (
        lr,
        hr,
        _,
    ) in train_loader:

        lr = lr.to(
            DEVICE,
            non_blocking=True,
        )

        hr = hr.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        if USE_AMP:

            with torch.amp.autocast(
                "cuda"
            ):

                prediction = (
                    model(
                        lr
                    )
                )

                (
                    loss,
                    l1,
                    ssim_l,
                    grad,
                    color,
                ) = compute_loss(
                    prediction,
                    hr,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            prediction = (
                model(
                    lr
                )
            )

            (
                loss,
                l1,
                ssim_l,
                grad,
                color,
            ) = compute_loss(
                prediction,
                hr,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
        )

        running_l1 += (
            l1.item()
        )

        running_ssim += (
            ssim_l.item()
        )

        running_grad += (
            grad.item()
        )

        running_color += (
            color.item()
        )

        batches += 1

    # --------------------------------------------------------
    # Training averages
    # --------------------------------------------------------

    train_loss = (
        running_loss
        / batches
    )

    train_l1 = (
        running_l1
        / batches
    )

    train_ssim = (
        running_ssim
        / batches
    )

    train_grad = (
        running_grad
        / batches
    )

    train_color = (
        running_color
        / batches
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_loss,
        val_l1,
        val_ssim,
        val_grad,
        val_color,
    ) = validate()

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()

    print(
        f"Epoch "
        f"{epoch:03d}/{EPOCHS}"
    )

    print(
        f"Learning rate  : "
        f"{current_lr:.7f}"
    )

    print(
        f"Train loss     : "
        f"{train_loss:.6f}"
    )

    print(
        f"Train L1       : "
        f"{train_l1:.6f}"
    )

    print(
        f"Train SSIM     : "
        f"{train_ssim:.6f}"
    )

    print(
        f"Train Gradient : "
        f"{train_grad:.6f}"
    )

    print(
        f"Train Color    : "
        f"{train_color:.6f}"
    )

    print(
        f"Val loss       : "
        f"{val_loss:.6f}"
    )

    print(
        f"Val L1         : "
        f"{val_l1:.6f}"
    )

    print(
        f"Val SSIM       : "
        f"{val_ssim:.6f}"
    )

    print(
        f"Val Gradient   : "
        f"{val_grad:.6f}"
    )

    print(
        f"Val Color      : "
        f"{val_color:.6f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = (
            val_loss
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "val_loss":
                    val_loss,

                "features":
                    FEATURES,

                "num_blocks":
                    NUM_BLOCKS,

                "scale":
                    SCALE,
            },
            CHECKPOINT_PATH,
        )

        print(
            "✅ NEW BEST MODEL SAVED"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()

print("=" * 78)

print(
    "LOADING BEST MODEL"
)

print("=" * 78)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print()

print(
    "Best epoch:",
    checkpoint[
        "epoch"
    ],
)

print(
    "Best validation loss:",
    checkpoint[
        "val_loss"
    ],
)


# ============================================================
# FIND BENGALURU DEMO IMAGE
# ============================================================

# ------------------------------------------------------------
# Prefer test set
# ------------------------------------------------------------

candidate_dirs = [
    TEST_HR_DIR,
    VAL_HR_DIR,
    TRAIN_HR_DIR,
]

bengaluru_files = []

selected_source_dir = None

for candidate_dir in candidate_dirs:

    if candidate_dir.exists():

        files = sorted(
            candidate_dir.glob(
                "bengaluru_20260427_*.npy"
            )
        )

        if files:

            bengaluru_files = files

            selected_source_dir = (
                candidate_dir
            )

            break


# ------------------------------------------------------------
# Fallback recursive search
# ------------------------------------------------------------

if not bengaluru_files:

    bengaluru_files = sorted(
        DATA_ROOT.glob(
            "**/bengaluru_20260427_*.npy"
        )
    )

    if bengaluru_files:

        selected_source_dir = (
            bengaluru_files[
                0
            ].parent
        )


# ------------------------------------------------------------
# Error if nothing found
# ------------------------------------------------------------

if not bengaluru_files:

    raise RuntimeError(
        "\nNo Bengaluru HR patches found.\n"
        "Expected files matching:\n"
        "bengaluru_20260427_*.npy\n"
    )


# ============================================================
# SELECT DEMO PATCH
# ============================================================

selected_hr_path = (
    bengaluru_files[0]
)

print()

print(
    "Demo source directory:",
    selected_source_dir,
)

print(
    "Selected Bengaluru patch:",
    selected_hr_path.name,
)


# ============================================================
# LOAD HR
# ============================================================

hr = np.load(
    selected_hr_path
).astype(
    np.float32
)

if hr.shape != (
    256,
    256,
    3,
):

    raise RuntimeError(
        "Unexpected Bengaluru HR shape: "
        f"{hr.shape}"
    )

hr = np.clip(
    hr,
    0.0,
    1.0,
)


# ============================================================
# CREATE FINAL DEMO LR
# ============================================================
#
# IMPORTANT:
#
# Training:
#       HR
#        ↓
#      same type
#      of degradation
#        ↓
#       LR
#        ↓
#       CNN
#
# Demo:
#       HR
#        ↓
#      same type
#      of degradation
#        ↓
#       LR
#        ↓
#       CNN
#
# This removes the previous train/inference mismatch.
# ============================================================

np.random.seed(
    DEMO_SEED
)

demo_dataset = (
    SyntheticSatelliteDataset(
        selected_source_dir,
        training=False,
        deterministic=True,
    )
)

lr_tensor = (
    demo_dataset.degrade(
        hr,
        randomize=False,
    )
)

lr_tensor = (
    lr_tensor
    .unsqueeze(0)
)


# ============================================================
# CNN INFERENCE
# ============================================================

lr_input = (
    lr_tensor
    .to(DEVICE)
)

with torch.no_grad():

    if USE_AMP:

        with torch.amp.autocast(
            "cuda"
        ):

            sr_tensor = (
                model(
                    lr_input
                )
            )

    else:

        sr_tensor = (
            model(
                lr_input
            )
        )


# ============================================================
# CLAMP
# ============================================================

sr_tensor = torch.clamp(
    sr_tensor,
    0.0,
    1.0,
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

lr = (
    lr_tensor
    .squeeze(0)
    .cpu()
    .numpy()
    .transpose(1, 2, 0)
)

sr = (
    sr_tensor
    .squeeze(0)
    .float()
    .cpu()
    .numpy()
    .transpose(1, 2, 0)
)


# ============================================================
# BICUBIC BASELINE
# ============================================================

bicubic_tensor = (
    F.interpolate(
        lr_tensor,
        size=(
            256,
            256,
        ),
        mode="bicubic",
        align_corners=False,
    )
)

bicubic = (
    bicubic_tensor
    .squeeze(0)
    .cpu()
    .numpy()
    .transpose(1, 2, 0)
)

bicubic = np.clip(
    bicubic,
    0.0,
    1.0,
)


# ============================================================
# METRICS
# ============================================================

bicubic_psnr = (
    peak_signal_noise_ratio(
        hr,
        bicubic,
        data_range=1.0,
    )
)

bicubic_ssim = (
    structural_similarity(
        hr,
        bicubic,
        channel_axis=2,
        data_range=1.0,
    )
)

sr_psnr = (
    peak_signal_noise_ratio(
        hr,
        sr,
        data_range=1.0,
    )
)

sr_ssim = (
    structural_similarity(
        hr,
        sr,
        channel_axis=2,
        data_range=1.0,
    )
)


# ============================================================
# COLOR STATISTICS
# ============================================================

print()

print("=" * 78)

print(
    "COLOR STATISTICS"
)

print("=" * 78)

for index, channel in enumerate(
    ["R", "G", "B"]
):

    hr_mean = (
        hr[:, :, index]
        .mean()
    )

    sr_mean = (
        sr[:, :, index]
        .mean()
    )

    hr_std = (
        hr[:, :, index]
        .std()
    )

    sr_std = (
        sr[:, :, index]
        .std()
    )

    print()

    print(
        f"{channel} mean | "
        f"HR: {hr_mean:.5f} | "
        f"SR: {sr_mean:.5f} | "
        f"Diff: "
        f"{sr_mean - hr_mean:+.5f}"
    )

    print(
        f"{channel} std  | "
        f"HR: {hr_std:.5f} | "
        f"SR: {sr_std:.5f} | "
        f"Diff: "
        f"{sr_std - hr_std:+.5f}"
    )


# ============================================================
# SAVE NUMPY OUTPUTS
# ============================================================

np.save(
    OUTPUT_LR_PATH,
    lr.astype(
        np.float32
    ),
)

np.save(
    OUTPUT_SR_PATH,
    sr.astype(
        np.float32
    ),
)


# ============================================================
# DIRECT RGB DISPLAY
# ============================================================
#
# No independent percentile stretching.
#
# All images use their actual [0,1] values.
#
# This prevents the visualization from artificially changing
# the apparent colors between LR/SR/HR.
# ============================================================

lr_rgb = np.clip(
    lr,
    0.0,
    1.0,
)

sr_rgb = np.clip(
    sr,
    0.0,
    1.0,
)

hr_rgb = np.clip(
    hr,
    0.0,
    1.0,
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        18,
        6,
    ),
)


# ============================================================
# LOW RES
# ============================================================

axes[0].imshow(
    lr_rgb,
    interpolation="nearest",
)

axes[0].set_title(
    "LOW-RES INPUT\n"
    "64 × 64",
    fontsize=17,
    fontweight="bold",
)


# ============================================================
# CNN
# ============================================================

axes[1].imshow(
    sr_rgb,
    interpolation="nearest",
)

axes[1].set_title(
    "OUR CNN OUTPUT\n"
    "256 × 256\n"
    f"{sr_psnr:.2f} dB PSNR | "
    f"{sr_ssim:.3f} SSIM",
    fontsize=17,
    fontweight="bold",
)


# ============================================================
# HR
# ============================================================

axes[2].imshow(
    hr_rgb,
    interpolation="nearest",
)

axes[2].set_title(
    "HR TARGET\n"
    "256 × 256",
    fontsize=17,
    fontweight="bold",
)


# ============================================================
# REMOVE AXES
# ============================================================

for ax in axes:

    ax.axis(
        "off"
    )


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Bengaluru Satellite Super-Resolution Demo\n"
    "64 × 64 → 256 × 256",
    fontsize=20,
    fontweight="bold",
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=(
        0,
        0,
        1,
        0.91,
    )
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    OUTPUT_IMAGE_PATH,
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print()

print("=" * 78)

print(
    "BENGALURU DEMO COMPLETE"
)

print("=" * 78)

print()

print(
    "Patch:",
    selected_hr_path.name,
)

print(
    "LR shape:",
    lr.shape,
)

print(
    "SR shape:",
    sr.shape,
)

print(
    "HR shape:",
    hr.shape,
)

print()

print(
    "BICUBIC BASELINE"
)

print(
    f"PSNR : "
    f"{bicubic_psnr:.4f} dB"
)

print(
    f"SSIM : "
    f"{bicubic_ssim:.4f}"
)

print()

print(
    "OUR CNN"
)

print(
    f"PSNR : "
    f"{sr_psnr:.4f} dB"
)

print(
    f"SSIM : "
    f"{sr_ssim:.4f}"
)

print()

print(
    "IMPROVEMENT"
)

print(
    f"PSNR : "
    f"{sr_psnr - bicubic_psnr:+.4f} dB"
)

print(
    f"SSIM : "
    f"{sr_ssim - bicubic_ssim:+.4f}"
)

print()

print(
    "CHECKPOINT:"
)

print(
    CHECKPOINT_PATH
)

print()

print(
    "VISUALIZATION:"
)

print(
    OUTPUT_IMAGE_PATH
)

print()

print(
    "LR ARRAY:"
)

print(
    OUTPUT_LR_PATH
)

print()

print(
    "SR ARRAY:"
)

print(
    OUTPUT_SR_PATH
)

print()

print(
    "✅ COMPLETE"
)