"""
Convolutional Neural Network for CIFAR-100 image classification.

Input:
    RGB images with shape (batch_size, 3, 32, 32)

Output:
    Raw class scores for 100 categories.
"""

import torch
import torch.nn as nn


class CIFAR100CNN(nn.Module):
    """
    A convolutional neural network for CIFAR-100 classification.
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()

        # Feature extraction layers.
        self.features = nn.Sequential(
            # Block 1: 32 x 32 images
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.10),

            # Block 2: 16 x 16 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.15),

            # Block 3: 8 x 8 feature maps
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.20),
        )

        # Classification layers.
        # After three pooling operations, the feature maps are 4 x 4.
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.40),

            nn.Linear(512, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw scores for every class."""

        features = self.features(images)
        scores = self.classifier(features)

        return scores


if __name__ == "__main__":
    # Quick architecture demonstration.
    model = CIFAR100CNN(num_classes=100)
    sample_images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        predictions = model(sample_images)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("CNN model created successfully!")
    print(f"Input shape: {tuple(sample_images.shape)}")
    print(f"Output shape: {tuple(predictions.shape)}")
    print(f"Number of classes: {predictions.shape[1]}")
    print(f"Trainable parameters: {parameter_count:,}")