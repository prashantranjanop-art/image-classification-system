"""
Evaluate the trained CIFAR-100 CNN on the independent test set.

Calculates:
- Test accuracy
- Macro and weighted precision, recall, and F1-score
- Per-class classification metrics
- Confusion matrix

All evaluation results are saved in reports/experiments/.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm


# --------------------------------------------------
# 1. Project paths and imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import build_dataloaders
from src.models.cnn_model import CIFAR100CNN


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pytorch"
    / "cifar100_cnn_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments"


# --------------------------------------------------
# 2. Evaluation function
# --------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
):
    """Generate predictions for every test image."""

    model.eval()

    true_labels = []
    predicted_labels = []

    print("\nEvaluating the independent test dataset...")

    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device, non_blocking=True)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(
            predictions.cpu().tolist()
        )

    return true_labels, predicted_labels


# --------------------------------------------------
# 3. Calculate classification metrics
# --------------------------------------------------

def calculate_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
) -> dict:
    """Calculate overall and per-class metrics."""

    test_accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            average="weighted",
            zero_division=0,
        )
    )

    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1_score": float(weighted_f1),
        "total_test_images": len(true_labels),
        "number_of_classes": len(class_names),
    }

    # Calculate detailed metrics for each class.
    per_class_precision, per_class_recall, per_class_f1, supports = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            average=None,
            zero_division=0,
        )
    )

    per_class_df = pd.DataFrame({
        "class_name": class_names,
        "precision": per_class_precision,
        "recall": per_class_recall,
        "f1_score": per_class_f1,
        "support": supports,
    })

    per_class_df.to_csv(
        OUTPUT_DIR / "cifar100_per_class_metrics.csv",
        index=False,
    )

    # Save the complete classification report.
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_path = OUTPUT_DIR / "cifar100_classification_report.txt"

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return metrics


# --------------------------------------------------
# 4. Generate confusion matrix
# --------------------------------------------------

def save_confusion_matrix(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
) -> None:
    """Save the confusion matrix as CSV and PNG."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_df.index.name = "Actual class"

    matrix_df.to_csv(
        OUTPUT_DIR / "cifar100_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(18, 16))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(
        "CIFAR-100 CNN: Test Confusion Matrix",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    positions = range(len(class_names))

    ax.set_xticks(positions)
    ax.set_yticks(positions)

    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=5,
    )

    ax.set_yticklabels(
        class_names,
        fontsize=5,
    )

    fig.tight_layout()

    figure_path = (
        OUTPUT_DIR / "cifar100_confusion_matrix.png"
    )

    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nConfusion matrix image: {figure_path}")
    print(
        "Confusion matrix CSV: "
        f"{OUTPUT_DIR / 'cifar100_confusion_matrix.csv'}"
    )


# --------------------------------------------------
# 5. Main evaluation workflow
# --------------------------------------------------

def main() -> None:
    """Load the best checkpoint and evaluate the test set."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Train the model before running evaluation."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("CIFAR-100 CNN TEST EVALUATION")
    print("=" * 65)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load the best validation checkpoint.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    # Load the data. Only the independent test loader is used
    # for final performance measurement.
    _, _, test_loader, class_names = build_dataloaders(
        batch_size=128,
        validation_split=0.10,
        random_seed=42,
    )

    model = CIFAR100CNN(
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"Best checkpoint: {MODEL_PATH}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.2f}%"
    )
    print(f"Test images: {len(test_loader.dataset):,}")

    # Predict on all test images.
    true_labels, predicted_labels = evaluate_model(
        model,
        test_loader,
        device,
    )

    # Calculate and display metrics.
    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
        class_names,
    )

    metrics_path = OUTPUT_DIR / "cifar100_test_metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    print(
        f"Test accuracy: "
        f"{metrics['test_accuracy_percent']:.2f}%"
    )
    print(
        f"Macro precision: "
        f"{metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro recall: "
        f"{metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1-score: "
        f"{metrics['macro_f1_score']:.4f}"
    )
    print(
        f"Weighted precision: "
        f"{metrics['weighted_precision']:.4f}"
    )
    print(
        f"Weighted recall: "
        f"{metrics['weighted_recall']:.4f}"
    )
    print(
        f"Weighted F1-score: "
        f"{metrics['weighted_f1_score']:.4f}"
    )

    print("\nGenerating confusion matrix...")

    save_confusion_matrix(
        true_labels,
        predicted_labels,
        class_names,
    )

    print(
        "\nDetailed classification report: "
        f"{OUTPUT_DIR / 'cifar100_classification_report.txt'}"
    )

    print(
        "Per-class metrics: "
        f"{OUTPUT_DIR / 'cifar100_per_class_metrics.csv'}"
    )

    print(f"Metrics JSON: {metrics_path}")

    print("\nTest evaluation completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()