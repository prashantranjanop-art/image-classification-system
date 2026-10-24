# Dataset Storage

This directory stores dataset information and local dataset files.

## Current Dataset

CIFAR-100 is the initial dataset for the image classification project.

It contains:
- 60,000 color images
- 100 image classes
- 50,000 training images
- 10,000 test images
- Image dimensions of 32 x 32 pixels

## Directory Structure

- `raw/`: Original downloaded datasets
- `processed/`: Prepared datasets, if required

## Download Instructions

Run the dataset download script:

```powershell
python src/data/download_cifar100.py