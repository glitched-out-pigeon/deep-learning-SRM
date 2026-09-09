import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SEN2NAIPDataset(Dataset):
    """
    SEN2NAIP real Sentinel-2 -> NAIP dataset.

    LR:
        H x W x 4
        10 m

    HR:
        4H x 4W x 4
        2.5 m
    """

    def __init__(
        self,
        root,
        split="train",
        lr_crop_size=64,
        training=True,
    ):
        self.root = Path(root)
        self.split = split
        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = lr_crop_size * 4
        self.training = training

        self.lr_dir = self.root / split / "LR"
        self.hr_dir = self.root / split / "HR"

        self.lr_files = sorted(self.lr_dir.glob("*.npy"))

        if not self.lr_files:
            raise RuntimeError(
                f"No LR files found in {self.lr_dir}"
            )

        # Verify corresponding HR files exist.
        for lr_file in self.lr_files:
            hr_file = self.hr_dir / lr_file.name

            if not hr_file.exists():
                raise RuntimeError(
                    f"Missing HR pair for {lr_file.name}"
                )

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):

        lr_path = self.lr_files[index]
        hr_path = self.hr_dir / lr_path.name

        lr = np.load(lr_path).astype(np.float32)
        hr = np.load(hr_path).astype(np.float32)

        # ----------------------------------------------------
        # Verify dimensions
        # ----------------------------------------------------

        if lr.shape[2] != 4:
            raise RuntimeError(
                f"Expected 4 LR channels, got {lr.shape}"
            )

        if hr.shape[2] != 4:
            raise RuntimeError(
                f"Expected 4 HR channels, got {hr.shape}"
            )

        # ----------------------------------------------------
        # Random aligned crop
        # ----------------------------------------------------

        h_lr, w_lr, _ = lr.shape

        if h_lr < self.lr_crop_size or w_lr < self.lr_crop_size:
            raise RuntimeError(
                f"LR image too small: {lr.shape}"
            )

        max_y = h_lr - self.lr_crop_size
        max_x = w_lr - self.lr_crop_size

        if self.training:

            y = random.randint(0, max_y)
            x = random.randint(0, max_x)

        else:

            # Deterministic center crop for validation/test.
            y = max_y // 2
            x = max_x // 2

        hr_y = y * 4
        hr_x = x * 4

        lr = lr[
            y:y + self.lr_crop_size,
            x:x + self.lr_crop_size,
            :
        ]

        hr = hr[
            hr_y:hr_y + self.hr_crop_size,
            hr_x:hr_x + self.hr_crop_size,
            :
        ]

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.training:

            if random.random() < 0.5:
                lr = np.flip(lr, axis=1).copy()
                hr = np.flip(hr, axis=1).copy()

            if random.random() < 0.5:
                lr = np.flip(lr, axis=0).copy()
                hr = np.flip(hr, axis=0).copy()

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        lr = torch.from_numpy(
            lr.transpose(2, 0, 1)
        ).float()

        hr = torch.from_numpy(
            hr.transpose(2, 0, 1)
        ).float()

        return {
            "lr": lr,
            "hr": hr,
            "name": lr_path.stem,
        }