"""
Download and inspect the CIFAR-100 dataset.

Dataset files are stored locally and are not uploaded to GitHub.
"""

from pathlib import Path

from torchvision.datasets import CIFAR100


# Locate the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Store downloaded data inside data/raw/cifar100.
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cifar100"


def main() -> None:
    """Download CIFAR-100 and display basic dataset information."""

    print("=" * 60)
    print("CIFAR-100 DATASET DOWNLOAD")
    print("=" * 60)

    print(f"\nDataset location: {DATA_ROOT}")
    print("\nDownloading the training dataset...")

    train_dataset = CIFAR100(
        root=str(DATA_ROOT),
        train=True,
        download=True,
    )

    print("\nDownloading the test dataset...")

    test_dataset = CIFAR100(
        root=str(DATA_ROOT),
        train=False,
        download=True,
    )

    # Inspect one sample image.
    sample_image, sample_label = train_dataset[0]

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Training images : {len(train_dataset):,}")
    print(f"Test images     : {len(test_dataset):,}")
    print(f"Total images    : {len(train_dataset) + len(test_dataset):,}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    print(f"\nImage dimensions: {sample_image.size}")
    print(f"Image format    : {sample_image.mode}")
    print(f"Sample label ID : {sample_label}")
    print(f"Sample class    : {train_dataset.classes[sample_label]}")

    print("\nFirst 10 classes:")

    for index, class_name in enumerate(train_dataset.classes[:10]):
        print(f"{index:2d}. {class_name}")

    print("\nDataset downloaded and inspected successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()