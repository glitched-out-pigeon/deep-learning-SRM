import sys
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
from scripts.losses_real import SRLoss


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_clean"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "real_clean"
    / "clean_spectral_mapping.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "real_clean"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "sen2naip_clean_harmonized_best.pth"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

CROP_SIZE = 64

FEATURES = 64
NUM_BLOCKS = 8

NUM_WORKERS = 0

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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD CLEAN MAPPING
# ============================================================

if not MAPPING_FILE.exists():

    raise FileNotFoundError(
        f"Clean spectral mapping not found:\n"
        f"{MAPPING_FILE}\n\n"
        f"Run fit_clean_spectral_mapping.py first."
    )


with open(
    MAPPING_FILE,
    "r",
    encoding="utf-8",
) as f:

    mapping = json.load(f)


slopes = np.asarray(
    mapping["slopes"],
    dtype=np.float32,
)

intercepts = np.asarray(
    mapping["intercepts"],
    dtype=np.float32,
)


if len(slopes) != 4:
    raise RuntimeError(
        "Expected 4 spectral slopes."
    )

if len(intercepts) != 4:
    raise RuntimeError(
        "Expected 4 spectral intercepts."
    )


# ============================================================
# DATASET
# ============================================================

class CleanHarmonizedSEN2NAIPDataset(Dataset):
    """
    Leak-free SEN2NAIP dataset.

    Input:
        Harmonized Sentinel-2
        10 m
        4 bands

    Target:
        NAIP
        2.5 m
        4 bands

    Random aligned crops are used only during training.
    Validation uses deterministic center crops.
    """

    def __init__(
        self,
        root,
        split,
        crop_size=64,
        training=False,
    ):

        self.root = Path(root)
        self.split = split

        self.crop_size = crop_size
        self.hr_crop_size = crop_size * 4

        self.training = training

        self.lr_dir = (
            self.root
            / split
            / "LR"
        )

        self.hr_dir = (
            self.root
            / split
            / "HR"
        )

        self.files = sorted(
            self.lr_dir.glob("*.npy")
        )

        if not self.files:

            raise RuntimeError(
                f"No LR files found:\n"
                f"{self.lr_dir}"
            )

        # Check pairing
        for lr_file in self.files:

            hr_file = (
                self.hr_dir
                / lr_file.name
            )

            if not hr_file.exists():

                raise RuntimeError(
                    f"Missing HR pair:\n"
                    f"{hr_file}"
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        lr_path = self.files[index]

        hr_path = (
            self.hr_dir
            / lr_path.name
        )

        lr = np.load(
            lr_path
        ).astype(np.float32)

        hr = np.load(
            hr_path
        ).astype(np.float32)

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if lr.shape != (
            121,
            121,
            4,
        ):

            raise RuntimeError(
                f"Unexpected LR shape: "
                f"{lr.shape}"
            )

        if hr.shape != (
            484,
            484,
            4,
        ):

            raise RuntimeError(
                f"Unexpected HR shape: "
                f"{hr.shape}"
            )

        # ----------------------------------------------------
        # APPLY CLEAN TRAINING-ONLY MAPPING
        # ----------------------------------------------------

        lr = (
            lr
            * slopes.reshape(1, 1, 4)
            + intercepts.reshape(1, 1, 4)
        )

        lr = np.clip(
            lr,
            0.0,
            1.0,
        ).astype(np.float32)

        # ----------------------------------------------------
        # RANDOM ALIGNED CROP
        # ----------------------------------------------------

        max_y = (
            lr.shape[0]
            - self.crop_size
        )

        max_x = (
            lr.shape[1]
            - self.crop_size
        )

        if self.training:

            y = np.random.randint(
                0,
                max_y + 1,
            )

            x = np.random.randint(
                0,
                max_x + 1,
            )

        else:

            y = max_y // 2
            x = max_x // 2

        hr_y = y * 4
        hr_x = x * 4

        lr = lr[
            y:y + self.crop_size,
            x:x + self.crop_size,
            :
        ]

        hr = hr[
            hr_y:hr_y + self.hr_crop_size,
            hr_x:hr_x + self.hr_crop_size,
            :
        ]

        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        if self.training:

            # Horizontal flip
            if np.random.random() < 0.5:

                lr = np.flip(
                    lr,
                    axis=1,
                ).copy()

                hr = np.flip(
                    hr,
                    axis=1,
                ).copy()

            # Vertical flip
            if np.random.random() < 0.5:

                lr = np.flip(
                    lr,
                    axis=0,
                ).copy()

                hr = np.flip(
                    hr,
                    axis=0,
                ).copy()

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        lr = torch.from_numpy(
            lr.transpose(2, 0, 1)
        ).float()

        hr = torch.from_numpy(
            hr.transpose(2, 0, 1)
        ).float()

        return lr, hr


# ============================================================
# DATASETS
# ============================================================

train_dataset = (
    CleanHarmonizedSEN2NAIPDataset(
        root=DATA_ROOT,
        split="train",
        crop_size=CROP_SIZE,
        training=True,
    )
)

val_dataset = (
    CleanHarmonizedSEN2NAIPDataset(
        root=DATA_ROOT,
        split="val",
        crop_size=CROP_SIZE,
        training=False,
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
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()

if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("SEN2NAIP CLEAN / LEAK-FREE TRAINING")
print("=" * 70)

print("\nDevice:")
print(device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    props = torch.cuda.get_device_properties(0)

    print(
        "VRAM:",
        f"{props.total_memory / (1024 ** 3):.2f} GB"
    )

print("\nDataset:")
print(
    "Train:",
    len(train_dataset)
)

print(
    "Val:",
    len(val_dataset)
)

print("\nModel:")
print(
    "Input channels:",
    4
)

print(
    "Output channels:",
    4
)

print(
    "Features:",
    FEATURES
)

print(
    "Residual blocks:",
    NUM_BLOCKS
)

print(
    "Parameters:",
    f"{parameter_count:,}"
)

print("\nTraining:")
print(
    "Epochs:",
    NUM_EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Crop:",
    f"{CROP_SIZE} → {CROP_SIZE * 4}"
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print("\nClean spectral mapping:")

for i in range(4):

    print(
        f"Band {i}: "
        f"y = "
        f"{slopes[i]:.6f}x "
        f"+ {intercepts[i]:.6f}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_gradient = 0.0

    count = 0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(
                device,
                non_blocking=True,
            )

            hr = hr.to(
                device,
                non_blocking=True,
            )

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    sr = model(lr)

                    loss, l1, gradient = (
                        criterion(
                            sr,
                            hr,
                        )
                    )

            else:

                sr = model(lr)

                loss, l1, gradient = (
                    criterion(
                        sr,
                        hr,
                    )
                )

            total_loss += loss.item()
            total_l1 += l1.item()
            total_gradient += (
                gradient.item()
            )

            count += 1

    return (
        total_loss / count,
        total_l1 / count,
        total_gradient / count,
    )


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float(
    "inf"
)


print("\n" + "=" * 70)
print("STARTING CLEAN TRAINING")
print("=" * 70)


for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    model.train()

    running_loss = 0.0
    running_l1 = 0.0
    running_gradient = 0.0

    progress = tqdm(
        train_loader,
        desc=(
            f"Epoch "
            f"{epoch}/{NUM_EPOCHS}"
        ),
    )

    for lr, hr in progress:

        lr = lr.to(
            device,
            non_blocking=True,
        )

        hr = hr.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                sr = model(lr)

                loss, l1, gradient = (
                    criterion(
                        sr,
                        hr,
                    )
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

            sr = model(lr)

            loss, l1, gradient = (
                criterion(
                    sr,
                    hr,
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        running_loss += (
            loss.item()
        )

        running_l1 += (
            l1.item()
        )

        running_gradient += (
            gradient.item()
        )

        progress.set_postfix(
            loss=f"{loss.item():.5f}"
        )

    # --------------------------------------------------------
    # Average training statistics
    # --------------------------------------------------------

    batches = len(
        train_loader
    )

    train_loss = (
        running_loss / batches
    )

    train_l1 = (
        running_l1 / batches
    )

    train_gradient = (
        running_gradient / batches
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_loss,
        val_l1,
        val_gradient,
    ) = validate()

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.6f}"
    )

    print(
        f"Train L1: "
        f"{train_l1:.6f}"
    )

    print(
        f"Train Gradient: "
        f"{train_gradient:.6f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.6f}"
    )

    print(
        f"Val L1: "
        f"{val_l1:.6f}"
    )

    print(
        f"Val Gradient: "
        f"{val_gradient:.6f}"
    )

    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_loss":
                    val_loss,
                "parameters":
                    parameter_count,
                "slopes":
                    slopes.tolist(),
                "intercepts":
                    intercepts.tolist(),
            },
            BEST_CHECKPOINT,
        )

        print(
            "\n✅ New best CLEAN model saved!"
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
print("CLEAN TRAINING COMPLETE")
print("=" * 70)

print(
    "\nBest validation loss:",
    f"{best_val_loss:.6f}"
)

print(
    "\nBest checkpoint:",
    BEST_CHECKPOINT
)

print(
    "\n✅ Leakage-controlled training complete."
)