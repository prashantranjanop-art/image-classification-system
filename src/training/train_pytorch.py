"""
PyTorch training pipeline for CIFAR-100 image classification.

Features:
- GPU acceleration when available
- Training and validation metrics
- Learning-rate scheduling
- Best-model checkpointing
- Training-history CSV
- Resume training from the latest checkpoint
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import build_dataloaders
from src.models.cnn_model import CIFAR100CNN


CONFIG_PATH = PROJECT_ROOT / "configs" / "cifar100.yaml"

MODEL_DIR = PROJECT_ROOT / "models" / "pytorch"
REPORT_DIR = PROJECT_ROOT / "reports" / "experiments"

BEST_MODEL_PATH = MODEL_DIR / "cifar100_cnn_best.pth"
LAST_CHECKPOINT_PATH = MODEL_DIR / "cifar100_cnn_last.pth"

HISTORY_PATH = REPORT_DIR / "cifar100_training_history.csv"


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# 3. Training one epoch
# --------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{total_epochs} - Train",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / total_samples

    accuracy = (
        correct_predictions / total_samples
    ) * 100.0

    return average_loss, accuracy


# --------------------------------------------------
# 4. Validation
# --------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    epoch: int,
    total_epochs: int,
):
    """Evaluate the model without updating its parameters."""

    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{total_epochs} - Val",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples

    accuracy = (
        correct_predictions / total_samples
    ) * 100.0

    return average_loss, accuracy


# --------------------------------------------------
# 5. Save training history
# --------------------------------------------------

def save_history(history: list[dict]) -> None:
    """Save accumulated epoch metrics to a CSV file."""

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
    ]

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def load_history() -> list[dict]:
    """Load previously recorded history when resuming."""

    if not HISTORY_PATH.exists():
        return []

    with HISTORY_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# 6. Main training process
# --------------------------------------------------

def main() -> None:
    """Configure and run CNN training."""

    parser = argparse.ArgumentParser(
        description="Train the CIFAR-100 CNN."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total target epochs; defaults to the YAML configuration.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the configured batch size.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest training checkpoint.",
    )

    args = parser.parse_args()

    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive.")

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    # Read project settings.
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    training_config = config["training"]
    reproducibility_config = config.get(
        "reproducibility", {}
    )

    epochs = (
        args.epochs
        if args.epochs is not None
        else int(training_config["epochs"])
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(training_config["batch_size"])
    )

    learning_rate = float(
        training_config["learning_rate"]
    )

    validation_split = float(
        training_config["validation_split"]
    )

    seed = int(
        reproducibility_config.get("random_seed", 42)
    )

    set_seed(seed)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("CIFAR-100 CNN TRAINING")
    print("=" * 65)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Target epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    # Prepare data.
    (
        train_loader,
        validation_loader,
        test_loader,
        class_names,
    ) = build_dataloaders(
        batch_size=batch_size,
        validation_split=validation_split,
        random_seed=seed,
    )

    print(f"\nTraining images: {len(train_loader.dataset):,}")
    print(
        f"Validation images: "
        f"{len(validation_loader.dataset):,}"
    )
    print(f"Test images: {len(test_loader.dataset):,}")

    # Create the model.
    model = CIFAR100CNN(
        num_classes=len(class_names)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    start_epoch = 0
    best_val_accuracy = -1.0

    history = []

    # Resume from an earlier checkpoint if requested.
    if args.resume:
        if not LAST_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                "No latest checkpoint found. "
                "Run training without --resume first."
            )

        checkpoint = torch.load(
            LAST_CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = int(checkpoint["epoch"]) + 1

        best_val_accuracy = float(
            checkpoint["best_val_accuracy"]
        )

        history = load_history()

        print(
            f"\nResuming after epoch {start_epoch}."
        )

    if start_epoch >= epochs:
        print(
            f"\nTraining has already reached "
            f"the requested {epochs} epochs."
        )
        print(
            "Choose a larger --epochs value to continue."
        )
        return

    print("\nStarting training...\n")

    for epoch in range(start_epoch, epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            epochs,
        )

        val_loss, val_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            epoch,
            epochs,
        )

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Val loss:   {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.2f}%"
        )
        print(f"Learning rate: {current_lr:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "train_accuracy": f"{train_accuracy:.4f}",
            "val_loss": f"{val_loss:.6f}",
            "val_accuracy": f"{val_accuracy:.4f}",
            "learning_rate": f"{current_lr:.8f}",
        })

        save_history(history)

        # Save the best model based on validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "best_val_accuracy": best_val_accuracy,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"New best model saved: "
                f"{best_val_accuracy:.2f}% validation accuracy"
            )

        # Save the latest complete training state.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_accuracy": best_val_accuracy,
            },
            LAST_CHECKPOINT_PATH,
        )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(f"\nBest model: {BEST_MODEL_PATH}")
    print(f"Latest checkpoint: {LAST_CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")

    print(
        "\nThe independent test set has not been evaluated yet."
    )


if __name__ == "__main__":
    main()