import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from models.sr_cnn import ResidualPixelShuffleSR
from scripts.dataset_real import SEN2NAIPDataset
from scripts.losses_real import SRLoss


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "real"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "sen2naip_pixelshuffle_best.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

CROP_SIZE = 64

FEATURES = 64

NUM_BLOCKS = 8

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CUDA OPTIMIZATION
# ============================================================

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True


print("=" * 70)
print("SEN2NAIP REAL-DATA TRAINING")
print("=" * 70)

print("\nDevice:")
print(device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    gpu_properties = torch.cuda.get_device_properties(0)

    print(
        "VRAM:",
        f"{gpu_properties.total_memory / (1024 ** 3):.2f} GB"
    )


# ============================================================
# DATASETS
# ============================================================

train_dataset = SEN2NAIPDataset(
    root=DATA_ROOT,
    split="train",
    lr_crop_size=CROP_SIZE,
    training=True,
)

val_dataset = SEN2NAIPDataset(
    root=DATA_ROOT,
    split="val",
    lr_crop_size=CROP_SIZE,
    training=False,
)


print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Val:", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# MODEL
# ============================================================

model = ResidualPixelShuffleSR(
    in_channels=4,
    out_channels=4,
    features=FEATURES,
    num_blocks=NUM_BLOCKS,
).to(device)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print("\nModel:")
print("Input channels:", 4)
print("Output channels:", 4)
print("Features:", FEATURES)
print("Residual blocks:", NUM_BLOCKS)
print("Parameters:", f"{parameter_count:,}")


# ============================================================
# LOSS
# ============================================================

criterion = SRLoss(
    l1_weight=0.8,
    gradient_weight=0.2,
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# AMP
# ============================================================

use_amp = torch.cuda.is_available()

if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_gradient = 0.0

    count = 0

    with torch.no_grad():

        for batch in val_loader:

            lr = batch["lr"].to(
                device,
                non_blocking=True
            )

            hr = batch["hr"].to(
                device,
                non_blocking=True
            )

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    sr = model(lr)

                    loss, l1, gradient = criterion(
                        sr,
                        hr
                    )

            else:

                sr = model(lr)

                loss, l1, gradient = criterion(
                    sr,
                    hr
                )

            total_loss += loss.item()
            total_l1 += l1.item()
            total_gradient += gradient.item()

            count += 1

    return (
        total_loss / count,
        total_l1 / count,
        total_gradient / count,
    )


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(1, NUM_EPOCHS + 1):

    model.train()

    running_loss = 0.0
    running_l1 = 0.0
    running_gradient = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{NUM_EPOCHS}",
        leave=True,
    )

    for batch in progress:

        lr = batch["lr"].to(
            device,
            non_blocking=True
        )

        hr = batch["hr"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward + loss
        # ----------------------------------------------------

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                sr = model(lr)

                loss, l1, gradient = criterion(
                    sr,
                    hr
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            sr = model(lr)

            loss, l1, gradient = criterion(
                sr,
                hr
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        running_loss += loss.item()
        running_l1 += l1.item()
        running_gradient += gradient.item()

        progress.set_postfix(
            loss=f"{loss.item():.5f}"
        )

    # --------------------------------------------------------
    # Average training loss
    # --------------------------------------------------------

    num_batches = len(train_loader)

    train_loss = (
        running_loss / num_batches
    )

    train_l1 = (
        running_l1 / num_batches
    )

    train_gradient = (
        running_gradient / num_batches
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_l1, val_gradient = validate()

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.6f}"
    )

    print(
        f"Train L1: {train_l1:.6f}"
    )

    print(
        f"Train Gradient: {train_gradient:.6f}"
    )

    print(
        f"Val Loss: {val_loss:.6f}"
    )

    print(
        f"Val L1: {val_l1:.6f}"
    )

    print(
        f"Val Gradient: {val_gradient:.6f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "parameters": parameter_count,
            },
            BEST_CHECKPOINT,
        )

        print(
            "\n✅ New best model saved!"
        )

        print(
            "Path:",
            BEST_CHECKPOINT
        )

    print("-" * 70)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "\nBest validation loss:",
    f"{best_val_loss:.6f}"
)

print(
    "\nBest checkpoint:",
    BEST_CHECKPOINT
)

print("\n✅ Real SEN2NAIP training finished.")