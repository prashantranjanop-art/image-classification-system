"""
Explore the CIFAR-100 dataset.

This script:
1. Loads the training dataset.
2. Displays sample images with their labels.
3. Visualizes the distribution of the 100 classes.
4. Saves the visualizations in reports/experiments/.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cifar100"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

def load_dataset() -> CIFAR100:
    """Load the locally downloaded CIFAR-100 training dataset."""

    print("Loading CIFAR-100 training dataset...")

    dataset = CIFAR100(
        root=str(DATA_ROOT),
        train=True,
        download=False,
    )

    print(f"Training samples: {len(dataset):,}")
    print(f"Number of classes: {len(dataset.classes)}")

    return dataset


# --------------------------------------------------
# 3. Display sample images
# --------------------------------------------------

def save_sample_images(dataset: CIFAR100) -> None:
    """Create a grid containing 20 labeled sample images."""

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))

    fig.suptitle(
        "CIFAR-100 Dataset: Sample Images",
        fontsize=16,
        fontweight="bold",
    )

    for index, ax in enumerate(axes.flat):
        image, label = dataset[index]

        ax.imshow(image)
        ax.set_title(dataset.classes[label], fontsize=9)
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = OUTPUT_DIR / "cifar100_sample_images.png"

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSample image visualization saved to:\n{output_path}")


# --------------------------------------------------
# 4. Analyze class distribution
# --------------------------------------------------

def save_class_distribution(dataset: CIFAR100) -> None:
    """Plot the number of training images in each class."""

    label_counts = Counter(dataset.targets)

    class_names = dataset.classes

    counts = [
        label_counts[class_id]
        for class_id in range(len(class_names))
    ]

    fig, ax = plt.subplots(figsize=(18, 7))

    ax.bar(range(len(class_names)), counts)

    ax.set_title(
        "CIFAR-100: Training Images per Class",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Image Classes")
    ax.set_ylabel("Number of Training Images")

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=7,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "cifar100_class_distribution.png"

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Class distribution saved to:\n{output_path}")

    print(f"\nMinimum images per class: {min(counts)}")
    print(f"Maximum images per class: {max(counts)}")
    print(f"Average images per class: {sum(counts) / len(counts):.0f}")


# --------------------------------------------------
# 5. Main function
# --------------------------------------------------

def main() -> None:
    """Run the dataset exploration workflow."""

    print("=" * 60)
    print("CIFAR-100 DATASET EXPLORATION")
    print("=" * 60)

    dataset = load_dataset()

    save_sample_images(dataset)

    save_class_distribution(dataset)

    print("\nDataset exploration completed successfully!")
    print(f"Results directory: {OUTPUT_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()