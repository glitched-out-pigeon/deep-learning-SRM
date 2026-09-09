from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_clean"
)


def inspect_split(split):

    lr_dir = ROOT / split / "LR"
    hr_dir = ROOT / split / "HR"

    lr_files = sorted(
        lr_dir.glob("*.npy")
    )

    hr_files = sorted(
        hr_dir.glob("*.npy")
    )

    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    print(
        "LR:",
        len(lr_files)
    )

    print(
        "HR:",
        len(hr_files)
    )

    assert len(lr_files) == len(
        hr_files
    )

    names_lr = {
        p.name
        for p in lr_files
    }

    names_hr = {
        p.name
        for p in hr_files
    }

    assert names_lr == names_hr

    sample = lr_files[0]

    lr = np.load(
        sample
    )

    hr = np.load(
        hr_dir / sample.name
    )

    print(
        "\nSample:",
        sample.name
    )

    print(
        "LR shape:",
        lr.shape
    )

    print(
        "HR shape:",
        hr.shape
    )

    print(
        "LR dtype:",
        lr.dtype
    )

    print(
        "HR dtype:",
        hr.dtype
    )

    print(
        "LR range:",
        f"{lr.min():.6f} -> {lr.max():.6f}"
    )

    print(
        "HR range:",
        f"{hr.min():.6f} -> {hr.max():.6f}"
    )

    assert lr.shape == (
        121,
        121,
        4,
    )

    assert hr.shape == (
        484,
        484,
        4,
    )

    assert lr.dtype == np.float32
    assert hr.dtype == np.float32

    assert 0.0 <= lr.min()
    assert lr.max() <= 1.0

    assert 0.0 <= hr.min()
    assert hr.max() <= 1.0

    print(
        "\n✅ Split validation passed."
    )


def main():

    print("=" * 70)
    print("LEAK-FREE REAL DATASET VALIDATION")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test",
    ]:

        inspect_split(
            split
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ ALL LEAK-FREE DATASET CHECKS PASSED."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()