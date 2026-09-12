"""
Quickstart: explain a single prediction with Grad-CAM.

Run with:  python examples/quickstart.py
"""
import sys
from pathlib import Path

# Ensure local lensight package is discoverable when running example directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import torch.nn as nn
from PIL import Image

from lensight import GradCAM, HiResCAM, ContrastiveCAM, IntegratedGradients
from lensight.utils.image_utils import denormalize, overlay_heatmap


class SimpleCNN(nn.Module):
    """Stand-in for your real model — swap this for a trained ResNet/EfficientNet/etc."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


def main():
    model = SimpleCNN(num_classes=10)
    model.eval()

    # In real usage this comes from your preprocessing pipeline
    # (e.g. torchvision.transforms.Compose([...])) applied to a real image.
    image = torch.randn(3, 224, 224)

    # --- Grad-CAM: where did the model look? ---
    cam = GradCAM(model)  # auto-detects the last Conv2d layer
    heatmap = cam.explain(image)  # predicted class by default
    rgb = denormalize(image)
    overlay = overlay_heatmap(rgb, heatmap)
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "gradcam_overlay.png")
    Image.fromarray(overlay).save(out_path)
    print(f"Saved {out_path}")

    # --- HiResCAM: elementwise gradient multiplication without spatial pooling ---
    hires = HiResCAM(model)
    hires_map = hires.explain(image)
    print(f"HiResCAM map range: [{hires_map.min():.2f}, {hires_map.max():.2f}]")

    # --- Contrastive CAM: why class A instead of runner-up class B? ---
    contrastive = ContrastiveCAM(model)
    contrast_map = contrastive.explain(image)
    print(f"Contrastive CAM map range: [{contrast_map.min():.2f}, {contrast_map.max():.2f}]")

    # --- Integrated Gradients: pixel-level attribution ---
    ig = IntegratedGradients(model, steps=32)
    saliency = ig.explain(image)
    print(f"Saliency map shape: {saliency.shape}, range [{saliency.min():.2f}, {saliency.max():.2f}]")


if __name__ == "__main__":
    main()
