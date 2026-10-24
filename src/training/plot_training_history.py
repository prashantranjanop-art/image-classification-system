"""
Plot training and validation performance for the CIFAR-100 CNN.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "cifar100_training_history.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments"


# --------------------------------------------------
# 2. Load training history
# --------------------------------------------------

def load_history() -> pd.DataFrame:
    """Load recorded training metrics from CSV."""

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}\n"
            "Train the CNN before generating the graphs."
        )

    history = pd.read_csv(HISTORY_PATH)

    required_columns = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    }

    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in training history: {missing_columns}"
        )

    if history.empty:
        raise ValueError("Training history is empty.")

    return history


# --------------------------------------------------
# 3. Plot accuracy
# --------------------------------------------------

def plot_accuracy(history: pd.DataFrame) -> None:
    """Save training and validation accuracy curves."""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        history["epoch"],
        history["train_accuracy"],
        marker="o",
        label="Training Accuracy",
    )

    ax.plot(
        history["epoch"],
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy",
    )

    ax.set_title("CIFAR-100 CNN: Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "cifar100_accuracy_curves.png"

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Accuracy graph saved to:\n{output_path}")


# --------------------------------------------------
# 4. Plot loss
# --------------------------------------------------

def plot_loss(history: pd.DataFrame) -> None:
    """Save training and validation loss curves."""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Training Loss",
    )

    ax.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation Loss",
    )

    ax.set_title("CIFAR-100 CNN: Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "cifar100_loss_curves.png"

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Loss graph saved to:\n{output_path}")


# --------------------------------------------------
# 5. Main function
# --------------------------------------------------

def main() -> None:
    """Generate the training performance graphs."""

    print("=" * 60)
    print("CIFAR-100 TRAINING PERFORMANCE VISUALIZATION")
    print("=" * 60)

    history = load_history()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nRecorded epochs: {len(history)}")

    plot_accuracy(history)
    plot_loss(history)

    best_index = history["val_accuracy"].idxmax()
    best_row = history.loc[best_index]

    print("\nBEST RECORDED VALIDATION RESULT")
    print("-" * 60)
    print(f"Epoch: {int(best_row['epoch'])}")
    print(f"Validation accuracy: {best_row['val_accuracy']:.2f}%")
    print(f"Validation loss: {best_row['val_loss']:.4f}")

    print("\nVisualization completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()