from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SuperResolutionDataset(Dataset):
    """
    PyTorch Dataset for paired LR/HR super-resolution data.

    LR shape on disk: H x W x 3
    HR shape on disk: H x W x 3

    Returned tensors:
        LR: 3 x H x W
        HR: 3 x H x W
    """

    def __init__(self, root_dir: str | Path, split: str):
        self.root_dir = Path(root_dir)
        self.split = split

        self.lr_dir = self.root_dir / split / "LR"
        self.hr_dir = self.root_dir / split / "HR"

        if not self.lr_dir.exists():
            raise FileNotFoundError(
                f"LR directory not found: {self.lr_dir}"
            )

        if not self.hr_dir.exists():
            raise FileNotFoundError(
                f"HR directory not found: {self.hr_dir}"
            )

        self.lr_files = sorted(self.lr_dir.glob("*.npy"))

        if not self.lr_files:
            raise RuntimeError(
                f"No .npy files found in {self.lr_dir}"
            )

        # Make sure every LR file has a matching HR file.
        for lr_path in self.lr_files:
            hr_path = self.hr_dir / lr_path.name

            if not hr_path.exists():
                raise RuntimeError(
                    f"Missing HR counterpart for: {lr_path.name}"
                )

        print(
            f"{split.upper()} dataset loaded: "
            f"{len(self.lr_files)} pairs"
        )

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, index: int):
        lr_path = self.lr_files[index]
        hr_path = self.hr_dir / lr_path.name

        # Load NumPy arrays.
        lr = np.load(lr_path).astype(np.float32)
        hr = np.load(hr_path).astype(np.float32)

        # Basic validation.
        if lr.ndim != 3 or lr.shape[2] != 3:
            raise ValueError(
                f"Invalid LR shape {lr.shape} in {lr_path.name}"
            )

        if hr.ndim != 3 or hr.shape[2] != 3:
            raise ValueError(
                f"Invalid HR shape {hr.shape} in {hr_path.name}"
            )

        # Convert HWC -> CHW.
        lr = np.transpose(lr, (2, 0, 1))
        hr = np.transpose(hr, (2, 0, 1))

        # Convert NumPy -> PyTorch tensors.
        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        return lr, hr


if __name__ == "__main__":
    DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"

    print("=" * 60)
    print("Testing SuperResolutionDataset")
    print("=" * 60)

    train_dataset = SuperResolutionDataset(
        root_dir=DATA_ROOT,
        split="train",
    )

    print(f"\nDataset length: {len(train_dataset)}")

    lr, hr = train_dataset[0]

    print("\nFirst sample:")
    print(f"LR shape:   {lr.shape}")
    print(f"LR dtype:   {lr.dtype}")
    print(f"LR range:   {lr.min():.6f} -> {lr.max():.6f}")

    print(f"\nHR shape:   {hr.shape}")
    print(f"HR dtype:   {hr.dtype}")
    print(f"HR range:   {hr.min():.6f} -> {hr.max():.6f}")

    # Verify expected dimensions.
    assert lr.shape == (3, 64, 64)
    assert hr.shape == (3, 256, 256)

    # Verify values are within our expected range.
    assert lr.min() >= 0.0
    assert lr.max() <= 1.0

    assert hr.min() >= 0.0
    assert hr.max() <= 1.0

    print("\n✅ Dataset test passed!")