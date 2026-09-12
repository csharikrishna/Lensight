"""
Full workflow: diagnose a trained classifier over a validation set and get
back a shareable HTML report with failure clusters, calibration, and
Grad-CAM overlays for the worst mistakes.

Run with:  python examples/full_diagnosis.py
Produces:  diagnosis_report.html  (open it in a browser)

This uses a synthetic dataset with a deliberately-injected bias — one class
is systematically confused with another — so you can see ModelDoctor
actually surface that failure mode rather than just dumping raw numbers.
"""
import sys
from pathlib import Path

# Ensure local lensight package is discoverable when running example directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import torch.nn as nn
import torch.utils.data as data

from lensight import ModelDoctor, TemperatureScaler

CLASS_NAMES = ["cat", "dog", "car", "truck", "bird"]
NUM_CLASSES = len(CLASS_NAMES)


class ToyClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


def build_biased_dataset(n=200, image_size=32, seed=0):
    """
    Each class gets a distinct mean pixel value so a model *can* learn to
    separate them — except classes 0 ("cat") and 1 ("dog") are given
    overlapping means, so a real model will systematically confuse them,
    the same way a real cat/dog classifier struggles on lookalike breeds.
    """
    g = torch.Generator().manual_seed(seed)
    class_means = torch.tensor([0.0, 0.05, 1.0, 1.5, -1.0])  # 0 & 1 overlap on purpose

    labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    images = torch.randn(n, 3, image_size, image_size, generator=g) * 0.5
    images += class_means[labels].view(-1, 1, 1, 1)
    return data.TensorDataset(images, labels)


def train_briefly(model, dataloader, epochs=8, lr=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs}  loss={total_loss / len(dataloader):.3f}")
    model.eval()


def main():
    train_ds = build_biased_dataset(n=300, seed=1)
    val_ds = build_biased_dataset(n=120, seed=2)

    train_loader = data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=16, shuffle=False)

    model = ToyClassifier()
    print("Training a small classifier on the biased toy dataset...")
    train_briefly(model, train_loader)

    print("\nRunning ModelDoctor diagnosis on the validation set...")
    doctor = ModelDoctor(model, class_names=CLASS_NAMES)
    report = doctor.diagnose(val_loader, n_clusters=4, n_example_images=6)

    print("\n" + report.summary_text())

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "diagnosis_report.html")
    report.save_html(report_path)
    print(f"\nSaved {report_path} — open it in a browser to see the "
          "confusion breakdown, calibration chart, and Grad-CAM overlays.")


if __name__ == "__main__":
    main()
