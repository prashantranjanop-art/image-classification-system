"""
Evaluate the refined ConvNeXt-Tiny model on CIFAR-100.

Reports:
- Test accuracy
- Macro precision, recall, and F1-score
- Weighted F1-score
- Per-class classification metrics
- Confusion matrix

This script evaluates the refined MixUp/CutMix checkpoint.
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

from src.data.resnet50_dataset import build_resnet50_dataloaders
from src.models.convnext_tiny_transfer import create_convnext_tiny


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pytorch"
    / "convnext_tiny_cifar100_refined_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments"


# --------------------------------------------------
# 2. Evaluate the model
# --------------------------------------------------

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """Generate predictions using horizontal-flip test-time augmentation."""

    model.eval()

    actual_labels = []
    predicted_labels = []

    use_amp = device.type == "cuda"

    for images, labels in tqdm(
        test_loader,
        desc="Evaluating test images",
    ):
        images = images.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            # Predict using the original images
            outputs = model(images)

            # Predict using horizontally flipped images
            flipped_images = torch.flip(images, dims=[3])
            flipped_outputs = model(flipped_images)

            # Average the predictions
            outputs = (outputs + flipped_outputs) / 2

        # Obtain the final predicted class for each image
        _, predictions = torch.max(outputs, dim=1)

        # Save predictions from EVERY batch
        actual_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.cpu().tolist())

    # Return after every batch has been processed
    return actual_labels, predicted_labels

# --------------------------------------------------
# 3. Calculate and save metrics
# --------------------------------------------------

def save_metrics(actual_labels, predicted_labels, class_names):
    """Calculate overall and per-class metrics."""

    labels = list(range(len(class_names)))

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual_labels,
            predicted_labels,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            actual_labels,
            predicted_labels,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )

    metrics = {
        "model": "ConvNeXt-Tiny refined",
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "best_validation_accuracy_percent": BEST_VALIDATION_ACCURACY,
        "test_images": len(actual_labels),
        "number_of_classes": len(class_names),
        "test_accuracy_percent": round(float(accuracy * 100), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1_score": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_precision), 4),
        "weighted_recall": round(float(weighted_recall), 4),
        "weighted_f1_score": round(float(weighted_f1), 4),
    }

    # Save overall metrics.
    metrics_path = (
        OUTPUT_DIR / "cifar100_convnext_refined_tta_test_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    # Save per-class metrics.
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual_labels,
            predicted_labels,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )

    per_class_df = pd.DataFrame({
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    })

    per_class_path = (
        OUTPUT_DIR / "cifar100_convnext_refined_tta_per_class.csv"
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    # Save the detailed classification report.
    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_path = (
        OUTPUT_DIR / "cifar100_convnext_refined_tta_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # Save confusion matrix data.
    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_df.index.name = "Actual class"

    matrix_path = (
        OUTPUT_DIR / "cifar100_convnext_refined_tta_confusion_matrix.csv"
    )

    matrix_df.to_csv(matrix_path)

    # Save a visual confusion matrix.
    fig, ax = plt.subplots(figsize=(18, 16))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(
        "refined ConvNeXt-Tiny: CIFAR-100 Test Confusion Matrix"
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
        OUTPUT_DIR / "cifar100_convnext_refined_tta_confusion_matrix.png"
    )

    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return metrics, metrics_path, per_class_path, report_path, figure_path


# --------------------------------------------------
# 4. Main evaluation workflow
# --------------------------------------------------

def main():
    """Load the best refined checkpoint and evaluate it."""

    global CHECKPOINT_EPOCH, BEST_VALIDATION_ACCURACY

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"refined model checkpoint not found: {MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("refined CONVNEXT-TINY TEST EVALUATION")
    print("=" * 65)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    CHECKPOINT_EPOCH = int(checkpoint["epoch"])
    BEST_VALIDATION_ACCURACY = float(
        checkpoint["best_val_accuracy"]
    )

    model = create_convnext_tiny(
        num_classes=len(class_names),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Checkpoint epoch: {CHECKPOINT_EPOCH}")
    print(
        f"Best validation accuracy: "
        f"{BEST_VALIDATION_ACCURACY:.2f}%"
    )

    # Use the same CIFAR-100 test preprocessing as our
    # previous ResNet-50 and ConvNeXt evaluations.
    _, _, test_loader, dataset_classes = (
        build_resnet50_dataloaders(
            batch_size=8,
            validation_split=0.10,
            random_seed=42,
        )
    )

    if class_names != dataset_classes:
        raise ValueError(
            "Checkpoint classes do not match dataset classes."
        )

    print(f"Test images: {len(test_loader.dataset):,}")
    print("\nEvaluating the refined model...")

    actual_labels, predicted_labels = evaluate_model(
        model,
        test_loader,
        device,
    )

    (
        metrics,
        metrics_path,
        per_class_path,
        report_path,
        figure_path,
    ) = save_metrics(
        actual_labels,
        predicted_labels,
        class_names,
    )

    print("\n" + "=" * 65)
    print("refined CONVNEXT-TINY TEST RESULTS")
    print("=" * 65)

    print(
        f"Test accuracy: "
        f"{metrics['test_accuracy_percent']:.2f}%"
    )

    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1_score']:.4f}")

    print(
        f"Weighted F1-score: "
        f"{metrics['weighted_f1_score']:.4f}"
    )

    print(f"\nMetrics JSON: {metrics_path}")
    print(f"Per-class metrics: {per_class_path}")
    print(f"Classification report: {report_path}")
    print(f"Confusion matrix: {figure_path}")

    print("\nEvaluation completed successfully!")
    print("=" * 65)


CHECKPOINT_EPOCH = 0
BEST_VALIDATION_ACCURACY = 0.0


if __name__ == "__main__":
    main()