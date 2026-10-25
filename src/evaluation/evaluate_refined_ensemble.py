"""
Compare ResNet-50 and refined ConvNeXt-Tiny
using the CIFAR-100 validation set.

Methods:
1. ResNet-50
2. refined ConvNeXt-Tiny
3. refined ConvNeXt with horizontal-flip TTA
4. ResNet-50 + refined ConvNeXt ensemble
5. Ensemble with horizontal-flip TTA

Only the validation split is used for model selection.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


# --------------------------------------------------
# 1. Project paths and imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.resnet50_dataset import build_resnet50_dataloaders
from src.models.resnet50_transfer import create_resnet50
from src.models.convnext_tiny_transfer import create_convnext_tiny


MODEL_DIR = PROJECT_ROOT / "models" / "pytorch"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments"

RESNET_PATH = MODEL_DIR / "resnet50_cifar100_best.pth"

CONVNEXT_PATH = (
    MODEL_DIR / "convnext_tiny_cifar100_refined_best.pth"
)


# --------------------------------------------------
# 2. Load the models
# --------------------------------------------------

def load_models(device):
    """Load both trained models and verify class ordering."""

    if not RESNET_PATH.exists():
        raise FileNotFoundError(
            f"ResNet-50 checkpoint not found: {RESNET_PATH}"
        )

    if not CONVNEXT_PATH.exists():
        raise FileNotFoundError(
            f"refined ConvNeXt checkpoint not found: {CONVNEXT_PATH}"
        )

    resnet_checkpoint = torch.load(
        RESNET_PATH,
        map_location="cpu",
        weights_only=True,
    )

    convnext_checkpoint = torch.load(
        CONVNEXT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    resnet_classes = resnet_checkpoint["class_names"]
    convnext_classes = convnext_checkpoint["class_names"]

    if resnet_classes != convnext_classes:
        raise ValueError(
            "The models have different class orderings."
        )

    resnet = create_resnet50(
        num_classes=len(resnet_classes),
        pretrained=False,
    )

    resnet.load_state_dict(
        resnet_checkpoint["model_state_dict"]
    )

    convnext = create_convnext_tiny(
        num_classes=len(convnext_classes),
        pretrained=False,
    )

    convnext.load_state_dict(
        convnext_checkpoint["model_state_dict"]
    )

    resnet = resnet.to(device).eval()
    convnext = convnext.to(device).eval()

    return resnet, convnext, resnet_classes


# --------------------------------------------------
# 3. Calculate class probabilities
# --------------------------------------------------

@torch.no_grad()
def get_probabilities(model, images, device):
    """Return softmax probabilities for each image."""

    use_amp = device.type == "cuda"

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        logits = model(images)

    return F.softmax(logits.float(), dim=1)


# --------------------------------------------------
# 4. Compare methods on validation images
# --------------------------------------------------

@torch.no_grad()
def evaluate_methods(
    resnet,
    convnext,
    validation_loader,
    device,
):
    """Measure validation accuracy for each method."""

    correct = {
        "ResNet-50": 0,
        "refined ConvNeXt-Tiny": 0,
        "refined ConvNeXt + TTA": 0,
        "ResNet-50 + refined ConvNeXt": 0,
        "Ensemble + TTA": 0,
    }

    total = 0

    for images, labels in tqdm(
        validation_loader,
        desc="Comparing validation predictions",
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Predictions for the original images.
        resnet_probs = get_probabilities(
            resnet,
            images,
            device,
        )

        convnext_probs = get_probabilities(
            convnext,
            images,
            device,
        )

        # Predictions for horizontally flipped images.
        flipped_images = torch.flip(
            images,
            dims=[3],
        )

        resnet_flipped = get_probabilities(
            resnet,
            flipped_images,
            device,
        )

        convnext_flipped = get_probabilities(
            convnext,
            flipped_images,
            device,
        )

        # Average original and flipped probabilities.
        resnet_tta = (
            resnet_probs + resnet_flipped
        ) / 2.0

        convnext_tta = (
            convnext_probs + convnext_flipped
        ) / 2.0

        # Average probability predictions from both models.
        ensemble_probs = (
            resnet_probs + convnext_probs
        ) / 2.0

        ensemble_tta = (
            resnet_tta + convnext_tta
        ) / 2.0

        predictions = {
            "ResNet-50": resnet_probs.argmax(dim=1),

            "refined ConvNeXt-Tiny": (
                convnext_probs.argmax(dim=1)
            ),

            "refined ConvNeXt + TTA": (
                convnext_tta.argmax(dim=1)
            ),

            "ResNet-50 + refined ConvNeXt": (
                ensemble_probs.argmax(dim=1)
            ),

            "Ensemble + TTA": (
                ensemble_tta.argmax(dim=1)
            ),
        }

        for method, predicted_labels in predictions.items():
            correct[method] += (
                predicted_labels == labels
            ).sum().item()

        total += labels.size(0)

    return {
        method: {
            "correct_predictions": count,
            "total_images": total,
            "accuracy_percent": round(
                100.0 * count / total,
                4,
            ),
        }
        for method, count in correct.items()
    }


# --------------------------------------------------
# 5. Main workflow
# --------------------------------------------------

def main():
    """Run the validation comparison."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("refined CIFAR-100 ENSEMBLE VALIDATION")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    resnet, convnext, class_names = load_models(device)

    (
        _train_loader,
        validation_loader,
        _test_loader,
        dataset_classes,
    ) = build_resnet50_dataloaders(
        batch_size=16,
        validation_split=0.10,
        random_seed=42,
    )

    if class_names != dataset_classes:
        raise ValueError(
            "Checkpoint class names do not match dataset classes."
        )

    results = evaluate_methods(
        resnet,
        convnext,
        validation_loader,
        device,
    )

    print("\nVALIDATION RESULTS")
    print("-" * 70)

    for method, metrics in results.items():
        print(
            f"{method:<38}"
            f"{metrics['accuracy_percent']:.2f}% "
            f"({metrics['correct_predictions']:,}/"
            f"{metrics['total_images']:,})"
        )

    best_method = max(
        results,
        key=lambda method: results[method]["accuracy_percent"],
    )

    best_accuracy = results[best_method]["accuracy_percent"]

    print("\nBEST VALIDATION METHOD")
    print("-" * 70)
    print(f"Method: {best_method}")
    print(f"Accuracy: {best_accuracy:.2f}%")

    json_path = (
        OUTPUT_DIR / "cifar100_refined_ensemble_validation.json"
    )

    json_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    csv_path = (
        OUTPUT_DIR / "cifar100_refined_ensemble_validation.csv"
    )

    pd.DataFrame([
        {
            "method": method,
            **metrics,
        }
        for method, metrics in results.items()
    ]).to_csv(
        csv_path,
        index=False,
    )

    print(f"\nJSON results: {json_path}")
    print(f"CSV results: {csv_path}")

    print("\nValidation comparison completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()