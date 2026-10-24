"""Tests for CIFAR-100 image preprocessing."""

import torch
from PIL import Image

from src.data.dataset_loader import get_transforms


def test_training_transform_shape():
    """Training preprocessing should return a 32x32 RGB tensor."""

    train_transform, _ = get_transforms()

    image = Image.new("RGB", (32, 32), color=(120, 80, 40))
    tensor = train_transform(image)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_evaluation_transform_shape():
    """Evaluation preprocessing should return the expected tensor."""

    _, evaluation_transform = get_transforms()

    image = Image.new("RGB", (32, 32), color=(120, 80, 40))
    tensor = evaluation_transform(image)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()