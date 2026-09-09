from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SuperResolutionDataset


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4


def main():
    print("=" * 60)
    print("Testing PyTorch DataLoader")
    print("=" * 60)

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = SuperResolutionDataset(
        root_dir=DATA_ROOT,
        split="train",
    )

    val_dataset = SuperResolutionDataset(
        root_dir=DATA_ROOT,
        split="val",
    )

    test_dataset = SuperResolutionDataset(
        root_dir=DATA_ROOT,
        split="test",
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")

    print("\nDataLoader sizes:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # --------------------------------------------------------
    # Test one training batch
    # --------------------------------------------------------

    lr_batch, hr_batch = next(iter(train_loader))

    print("\nFirst training batch:")
    print(f"LR shape: {lr_batch.shape}")
    print(f"HR shape: {hr_batch.shape}")

    print(f"LR dtype: {lr_batch.dtype}")
    print(f"HR dtype: {hr_batch.dtype}")

    print(
        f"LR range: {lr_batch.min():.6f} -> "
        f"{lr_batch.max():.6f}"
    )

    print(
        f"HR range: {hr_batch.min():.6f} -> "
        f"{hr_batch.max():.6f}"
    )

    # --------------------------------------------------------
    # Test GPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:")
    print(device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Move one batch to GPU.
    lr_gpu = lr_batch.to(device)
    hr_gpu = hr_batch.to(device)

    print("\nAfter moving to device:")
    print("LR device:", lr_gpu.device)
    print("HR device:", hr_gpu.device)

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert lr_batch.ndim == 4
    assert hr_batch.ndim == 4

    assert lr_batch.shape[1:] == (3, 64, 64)
    assert hr_batch.shape[1:] == (3, 256, 256)

    assert lr_gpu.device.type == device.type
    assert hr_gpu.device.type == device.type

    print("\n✅ DataLoader + GPU test passed!")


if __name__ == "__main__":
    main()