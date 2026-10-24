"""
CIFAR-100 data loading and preprocessing.

Creates training, validation, and testing DataLoaders
for the image classification project.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cifar100"

NUM_CLASSES = 100

DEFAULT_BATCH_SIZE = 128
DEFAULT_VALIDATION_SPLIT = 0.10
DEFAULT_RANDOM_SEED = 42


# CIFAR-100 normalization statistics.
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# --------------------------------------------------
# 2. Image transformations
# --------------------------------------------------

def get_transforms():
    """Create transformations for training and evaluation."""

    train_transform = transforms.Compose([
        # Randomly crop the image with padding.
        transforms.RandomCrop(32, padding=4),

        # Randomly flip images horizontally.
        transforms.RandomHorizontalFlip(),

        # Convert images into PyTorch tensors.
        transforms.ToTensor(),

        # Normalize RGB channels.
        transforms.Normalize(
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD,
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD,
        ),
    ])

    return train_transform, evaluation_transform


# --------------------------------------------------
# 3. Build DataLoaders
# --------------------------------------------------

def build_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    """
    Build training, validation, and test DataLoaders.

    Returns:
        train_loader
        validation_loader
        test_loader
        class_names
    """

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    if not 0 < validation_split < 1:
        raise ValueError(
            "Validation split must be between 0 and 1."
        )

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"CIFAR-100 dataset not found at {DATA_ROOT}. "
            "Run src/data/download_cifar100.py first."
        )

    train_transform, evaluation_transform = get_transforms()

    print("Loading CIFAR-100 datasets...")

    # Training dataset uses data augmentation.
    training_dataset = datasets.CIFAR100(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=train_transform,
    )

    # Validation uses the same original training images,
    # but without random augmentation.
    validation_dataset = datasets.CIFAR100(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=evaluation_transform,
    )

    # Official CIFAR-100 test dataset.
    test_dataset = datasets.CIFAR100(
        root=str(DATA_ROOT),
        train=False,
        download=False,
        transform=evaluation_transform,
    )

    # Create a reproducible random split.
    generator = torch.Generator().manual_seed(random_seed)

    shuffled_indices = torch.randperm(
        len(training_dataset),
        generator=generator,
    ).tolist()

    validation_size = int(
        len(training_dataset) * validation_split
    )

    validation_indices = shuffled_indices[:validation_size]
    training_indices = shuffled_indices[validation_size:]

    # Separate subsets share the same underlying image ordering.
    train_subset = Subset(
        training_dataset,
        training_indices,
    )

    validation_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    # Windows-safe default: no multiprocessing workers.
    num_workers = 0

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_names = training_dataset.classes

    return (
        train_loader,
        validation_loader,
        test_loader,
        class_names,
    )


# --------------------------------------------------
# 4. Test the data pipeline
# --------------------------------------------------

def main() -> None:
    """Load the dataset and inspect one batch."""

    print("=" * 60)
    print("CIFAR-100 DATA PIPELINE TEST")
    print("=" * 60)

    train_loader, validation_loader, test_loader, class_names = (
        build_dataloaders()
    )

    print("\nDATASET SUMMARY")
    print("-" * 60)

    print(f"Training images: {len(train_loader.dataset):,}")
    print(f"Validation images: {len(validation_loader.dataset):,}")
    print(f"Test images: {len(test_loader.dataset):,}")
    print(f"Number of classes: {len(class_names)}")

    # Inspect one training batch.
    images, labels = next(iter(train_loader))

    print("\nTRAINING BATCH")
    print("-" * 60)

    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Label batch shape: {tuple(labels.shape)}")
    print(f"Image tensor dtype: {images.dtype}")
    print(f"Label tensor dtype: {labels.dtype}")

    print(f"\nGPU available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    print("\nData pipeline is working successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()