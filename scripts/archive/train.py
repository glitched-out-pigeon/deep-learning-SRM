from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from losses import SRLoss
from torch.utils.data import DataLoader

from dataset import SuperResolutionDataset
from models.sr_cnn import SRCNN4x


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 100

NUM_WORKERS = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CNN SUPER-RESOLUTION TRAINING")
print("=" * 60)

print("\nDevice:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Dataset
# ============================================================

train_dataset = SuperResolutionDataset(
    root_dir=DATA_ROOT,
    split="train",
)

val_dataset = SuperResolutionDataset(
    root_dir=DATA_ROOT,
    split="val",
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# Model
# ============================================================

model = SRCNN4x().to(device)

print(
    "\nTrainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
)


# ============================================================
# Loss
# ============================================================

criterion = SRLoss(
    l1_weight=0.8,
    gradient_weight=0.2,
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=8,
    min_lr=1e-6,
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for lr, hr in train_loader:

        lr = lr.to(device)
        hr = hr.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        sr = model(lr)

        # Calculate loss
        loss = criterion(sr, hr)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)

            loss = criterion(sr, hr)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.6f} "
        f"| Val Loss: {val_loss:.6f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint_path = (
            CHECKPOINT_DIR / "pixelshuffle_cnn_best.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        print(
            f"  ✓ Saved checkpoint: {checkpoint_path.name}"
        )


print("\n" + "=" * 60)
print("TRAINING COMPLETE ✅")
print("=" * 60)

print("Best validation loss:", f"{best_val_loss:.6f}")