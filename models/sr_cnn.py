import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels: int = 64):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):

        return x + self.block(x)


# ============================================================
# REAL SEN2NAIP 4x SUPER-RESOLUTION CNN
# ============================================================

class ResidualPixelShuffleSR(nn.Module):
    """
    4-channel Sentinel-2 RGBN
    10 m -> 2.5 m
    4x super-resolution.

    Architecture:

        LR
         |
         v
      Encoder
         |
    Residual blocks
         |
    PixelShuffle x2
         |
    PixelShuffle x2
         |
    Learned residual
         |
         +---- Bicubic baseline
         |
         v
        SR
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: int = 64,
        num_blocks: int = 8,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # Residual feature processing
        # ----------------------------------------------------

        self.body = nn.Sequential(
            *[
                ResidualBlock(features)
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # Upscaling x2
        # ----------------------------------------------------

        self.up1 = nn.Sequential(

            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1,
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------
        # Upscaling x2 again
        # ----------------------------------------------------

        self.up2 = nn.Sequential(

            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1,
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------
        # Output residual
        # ----------------------------------------------------

        self.tail = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            scale_factor=4,
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        shallow = self.head(x)

        features = self.body(shallow)

        features = self.body_conv(features)

        features = features + shallow

        # ----------------------------------------------------
        # Upscale
        # ----------------------------------------------------

        features = self.up1(features)

        features = self.up2(features)

        # ----------------------------------------------------
        # Predict residual
        # ----------------------------------------------------

        residual = self.tail(features)

        # ----------------------------------------------------
        # Residual learning
        # ----------------------------------------------------

        sr = bicubic + residual

        return sr


# Backward-compatible alias
SRCNN4x = ResidualPixelShuffleSR