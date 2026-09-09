from pathlib import Path
import shutil

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "highres"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
PATCH_SIZE = 256
STRIDE = 128
SCALE = 4
TRAIN_SCENES = 3
VAL_SCENES = 1

# Degradation model parameters
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA = 1.0
NOISE_STD = 0.01  # relative to the [0,1] reflectance range


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    return np.clip(data, 0.0, 1.0)


def load_rgb(scene_dir: Path) -> np.ndarray:
    b02_path = scene_dir / "B02.tiff"
    b03_path = scene_dir / "B03.tiff"
    b04_path = scene_dir / "B04.tiff"
    for path in [b02_path, b03_path, b04_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required band: {path}")
    b02 = read_band(b02_path)
    b03 = read_band(b03_path)
    b04 = read_band(b04_path)
    if not (b02.shape == b03.shape == b04.shape):
        raise ValueError(f"Band dimension mismatch in {scene_dir.name}")
    return np.stack([b04, b03, b02], axis=-1).astype(np.float32)


def extract_patches(image: np.ndarray) -> list[np.ndarray]:
    height, width, channels = image.shape
    if height < PATCH_SIZE or width < PATCH_SIZE:
        raise ValueError(f"Scene is too small for {PATCH_SIZE}x{PATCH_SIZE}: {image.shape}")
    y_positions = list(range(0, height - PATCH_SIZE + 1, STRIDE))
    x_positions = list(range(0, width - PATCH_SIZE + 1, STRIDE))
    last_y = height - PATCH_SIZE
    last_x = width - PATCH_SIZE
    if y_positions[-1] != last_y:
        y_positions.append(last_y)
    if x_positions[-1] != last_x:
        x_positions.append(last_x)
    patches = []
    for y in y_positions:
        for x in x_positions:
            patches.append(image[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :])
    return patches


def gaussian_blur(tensor: torch.Tensor, kernel_size: int = BLUR_KERNEL_SIZE, sigma: float = BLUR_SIGMA) -> torch.Tensor:
    """Approximate sensor point-spread function with a Gaussian blur."""
    channels = tensor.shape[1]
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    grid = coords.pow(2).unsqueeze(0) + coords.pow(2).unsqueeze(1)
    kernel = torch.exp(-grid / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    return F.conv2d(tensor, kernel, padding=padding, groups=channels)


def create_lr(hr_patch: np.ndarray) -> np.ndarray:
    height, width, channels = hr_patch.shape
    tensor = torch.from_numpy(hr_patch.transpose(2, 0, 1)).unsqueeze(0)

    # 1. Blur — approximates the sensor's point-spread function
    blurred = gaussian_blur(tensor)

    # 2. Downsample — same bicubic operator as before
    lr_tensor = F.interpolate(
        blurred,
        size=(height // SCALE, width // SCALE),
        mode="bicubic",
        align_corners=False,
    )

    # 3. Sensor-like noise
    noise = torch.randn_like(lr_tensor) * NOISE_STD
    lr_tensor = lr_tensor + noise

    # 4. Light quantization — simulate limited radiometric precision
    lr = lr_tensor.squeeze(0).numpy().transpose(1, 2, 0)
    lr = np.clip(lr, 0.0, 1.0)
    lr = np.round(lr * 255.0) / 255.0

    return lr.astype(np.float32)


def save_pair(split: str, scene_name: str, patch_number: int, hr: np.ndarray, lr: np.ndarray) -> None:
    hr_dir = OUTPUT_ROOT / split / "HR"
    lr_dir = OUTPUT_ROOT / split / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{scene_name}_{patch_number:04d}.npy"
    np.save(hr_dir / filename, hr.astype(np.float32))
    np.save(lr_dir / filename, lr.astype(np.float32))


def main():
    print("=" * 70)
    print("MULTI-SCENE SYNTHETIC SR DATASET (realistic degradation)")
    print("=" * 70)
    scene_dirs = sorted([path for path in RAW_ROOT.iterdir() if path.is_dir()])
    if len(scene_dirs) < 5:
        raise RuntimeError(f"Expected at least 5 scenes, found {len(scene_dirs)}")
    print("\nScenes found:")
    for scene in scene_dirs:
        print(f"  - {scene.name}")

    train_scenes = scene_dirs[:TRAIN_SCENES]
    val_scenes = scene_dirs[TRAIN_SCENES:TRAIN_SCENES + VAL_SCENES]
    test_scenes = scene_dirs[TRAIN_SCENES + VAL_SCENES:]
    splits = {"train": train_scenes, "val": val_scenes, "test": test_scenes}

    print("\n" + "=" * 70)
    print("SCENE SPLIT")
    print("=" * 70)
    for split, scenes in splits.items():
        print(f"\n{split.upper()}:")
        for scene in scenes:
            print(f"  {scene.name}")

    print("\nClearing previous processed dataset...")
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_ROOT / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    total_counts = {"train": 0, "val": 0, "test": 0}
    for split, scenes in splits.items():
        print("\n" + "=" * 70)
        print(f"PROCESSING {split.upper()}")
        print("=" * 70)
        for scene_dir in scenes:
            print(f"\nScene: {scene_dir.name}")
            rgb = load_rgb(scene_dir)
            print(f"RGB shape: {rgb.shape}")
            print(f"RGB range: {rgb.min():.6f} -> {rgb.max():.6f}")
            patches = extract_patches(rgb)
            print(f"HR patches: {len(patches)}")
            for patch_number, hr in enumerate(patches):
                save_pair(split, scene_dir.name, patch_number, hr, create_lr(hr))
                total_counts[split] += 1

    print("\n" + "=" * 70)
    print("DATASET CREATED (realistic degradation) ✅")
    print("=" * 70)
    print(f"Train pairs: {total_counts['train']}")
    print(f"Val pairs:   {total_counts['val']}")
    print(f"Test pairs:  {total_counts['test']}")
    print("\nOutput:", OUTPUT_ROOT)
    print("\nEach pair:")
    print("  HR = 256 × 256 × 3")
    print("  LR =  64 ×  64 × 3")
    print("\nScene-level split:")
    print("  No scene appears in multiple splits.")
    print("\nDegradation model: Gaussian blur -> bicubic downsample -> noise -> light quantization")


if __name__ == "__main__":
    main()