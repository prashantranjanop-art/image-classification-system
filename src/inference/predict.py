"""
CIFAR-100 image prediction module.

Features:
- Loads the trained CNN checkpoint
- Predicts the class of an image
- Displays the top five predictions
- Reports confidence scores
- Supports custom images and CIFAR-100 test samples
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pytorch"
    / "cifar100_cnn_best.pth"
)

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cifar100"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_model import CIFAR100CNN


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_image_transform():
    """Prepare an image for the trained CNN."""

    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD,
        ),
    ])


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

def load_model(device: torch.device):
    """Load the best trained model checkpoint."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Train the CNN before running predictions."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    model = CIFAR100CNN(
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, class_names, checkpoint


# --------------------------------------------------
# 4. Predict an image
# --------------------------------------------------

@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    top_k: int = 5,
):
    """Return the top-k predicted classes and scores."""

    if top_k < 1 or top_k > len(class_names):
        raise ValueError(
            f"top_k must be between 1 and {len(class_names)}."
        )

    image_tensor = image_tensor.unsqueeze(0).to(device)

    outputs = model(image_tensor)

    # Convert raw model scores to probabilities.
    probabilities = F.softmax(outputs, dim=1)[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
    )

    results = []

    for probability, class_index in zip(
        top_probabilities.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        results.append({
            "class_name": class_names[class_index],
            "confidence": probability * 100,
        })

    return results


# --------------------------------------------------
# 5. Load a CIFAR-100 test sample
# --------------------------------------------------

def load_test_sample(sample_index: int):
    """Load a real image from the CIFAR-100 test dataset."""

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_ROOT}\n"
            "Run src/data/download_cifar100.py first."
        )

    transform = get_image_transform()

    test_dataset = datasets.CIFAR100(
        root=str(DATA_ROOT),
        train=False,
        download=False,
        transform=transform,
    )

    if not 0 <= sample_index < len(test_dataset):
        raise ValueError(
            f"Sample index must be between "
            f"0 and {len(test_dataset) - 1}."
        )

    image_tensor, true_label = test_dataset[sample_index]

    true_class = test_dataset.classes[true_label]

    return image_tensor, true_class


# --------------------------------------------------
# 6. Main prediction workflow
# --------------------------------------------------

def main():
    """Run image prediction from the command line."""

    parser = argparse.ArgumentParser(
        description="Predict image classes using the CIFAR-100 CNN."
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a custom image.",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="CIFAR-100 test image index when no image path is provided.",
    )

    args = parser.parse_args()

    if args.sample_index < 0:
        parser.error("--sample-index cannot be negative.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CIFAR-100 IMAGE CLASSIFICATION")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, class_names, checkpoint = load_model(device)

    print(f"Model checkpoint: {MODEL_PATH}")
    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.2f}%"
    )

    transform = get_image_transform()
    true_class = None

    # Option A: Predict a user-provided image.
    if args.image:
        image_path = Path(args.image).expanduser()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_tensor = transform(rgb_image)

        print(f"\nInput image: {image_path}")

    # Option B: Predict a CIFAR-100 test image.
    else:
        image_tensor, true_class = load_test_sample(
            args.sample_index
        )

        print(
            f"\nCIFAR-100 test sample: {args.sample_index}"
        )

    predictions = predict_image(
        model=model,
        image_tensor=image_tensor,
        class_names=class_names,
        device=device,
        top_k=5,
    )

    print("\nTOP 5 PREDICTIONS")
    print("-" * 60)

    for rank, prediction in enumerate(predictions, start=1):
        print(
            f"{rank}. {prediction['class_name']:<20}"
            f" {prediction['confidence']:.2f}%"
        )

    print("\nFINAL RESULT")
    print("-" * 60)

    print(
        f"Predicted class: "
        f"{predictions[0]['class_name']}"
    )

    print(
        f"Model confidence score: "
        f"{predictions[0]['confidence']:.2f}%"
    )

    if true_class is not None:
        print(f"Actual class: {true_class}")

        print(
            "Correct prediction: "
            f"{predictions[0]['class_name'] == true_class}"
        )

    print(
        "\nNote: Confidence scores are model estimates, "
        "not guarantees of correctness."
    )

    print("\nPrediction completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()