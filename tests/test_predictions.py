"""Tests for the image prediction functionality."""

import pytest
import torch
import torch.nn as nn

from src.inference.predict import predict_image


class DummyModel(nn.Module):
    """Small deterministic model for testing prediction behavior."""

    def forward(self, images):
        scores = torch.tensor(
            [0.1, 0.9, 0.3, 0.4],
            device=images.device,
        )

        return scores.unsqueeze(0).repeat(images.shape[0], 1)


def test_top_five_style_prediction_order():
    """Predictions should be ordered by decreasing confidence."""

    model = DummyModel()
    image = torch.zeros(3, 32, 32)

    class_names = [
        "apple",
        "bicycle",
        "mountain",
        "otter",
    ]

    results = predict_image(
        model=model,
        image_tensor=image,
        class_names=class_names,
        device=torch.device("cpu"),
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["class_name"] == "bicycle"

    confidence_scores = [
        result["confidence"]
        for result in results
    ]

    assert confidence_scores == sorted(
        confidence_scores,
        reverse=True,
    )


def test_prediction_rejects_invalid_top_k():
    """Invalid top_k values should raise an error."""

    model = DummyModel()
    image = torch.zeros(3, 32, 32)

    with pytest.raises(ValueError):
        predict_image(
            model=model,
            image_tensor=image,
            class_names=["apple", "bicycle", "mountain", "otter"],
            device=torch.device("cpu"),
            top_k=10,
        )