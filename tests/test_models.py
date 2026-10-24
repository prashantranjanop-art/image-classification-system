"""Tests for the CIFAR-100 CNN architecture."""

import torch

from src.models.cnn_model import CIFAR100CNN


def test_cnn_output_shape():
    """The CNN should return 100 scores for each input image."""

    model = CIFAR100CNN(num_classes=100)
    model.eval()

    images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        outputs = model(images)

    assert outputs.shape == (2, 100)


def test_cnn_outputs_are_finite():
    """The model should produce finite numerical outputs."""

    model = CIFAR100CNN(num_classes=100)
    model.eval()

    images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        outputs = model(images)

    assert torch.isfinite(outputs).all()
    