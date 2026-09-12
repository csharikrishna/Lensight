"""
Train a ResNet-18 model on Fashion-MNIST and run the complete Lensight diagnostic toolkit.

This script:
1. Loads Fashion-MNIST (10 real-world clothing categories).
2. Builds and trains a ResNet-18 model for 3 quick epochs.
3. Runs single-image explainability: Grad-CAM, HiResCAM, Contrastive CAM ("Why Pullover instead of Coat?").
4. Runs ModelDoctor across the validation set, discovering natural failure clusters and calibration error.
5. Performs Temperature Scaling calibration to remediate model overconfidence.
6. Exports an interactive HTML report and JSON metrics.

Run with:
    python examples/train_and_diagnose_resnet.py
"""

import sys
from pathlib import Path

# Ensure local lensight package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from lensight import (
    GradCAM,
    HiResCAM,
    ContrastiveCAM,
    ModelDoctor,
    TemperatureScaler,
)
from lensight.utils.image_utils import denormalize, overlay_heatmap

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_fashion_resnet(num_classes=10):
    """ResNet-18 adapted for 1-channel 28x28 / 32x32 images."""
    model = models.resnet18(weights=None)
    # Adapt first conv to 1-channel grayscale input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Preserve spatial resolution for 28x28 inputs
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    print("=" * 72)
    print("  LENSIGHT: TRAINING & DIAGNOSING RESNET-18 ON FASHION-MNIST")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # 1. Load Dataset
    print("\n[Step 1] Loading Fashion-MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=False, transform=transform
    )
    full_val = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=False, transform=transform
    )

    # Use a fast representative subset (1200 train, 300 val) for rapid training
    train_subset = data.Subset(full_train, range(1200))
    val_subset = data.Subset(full_val, range(300))

    train_loader = data.DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_subset, batch_size=32, shuffle=False)
    print(f"[OK] Loaded {len(train_subset)} training samples and {len(val_subset)} validation samples.")

    # 2. Build and Train ResNet-18
    print("\n[Step 2] Training ResNet-18...")
    model = build_fashion_resnet(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(dim=1) == lbls).sum().item()
            total += imgs.size(0)

        epoch_acc = correct / total
        epoch_loss = total_loss / total
        print(f"  Epoch {epoch + 1}/3 - Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.1%}")

    model.eval()
    print("[OK] ResNet-18 training complete.")

    # 3. Single-Image Attribution with CAM Explainers
    print("\n[Step 3] Visualizing Predictions with CAM Explainers...")
    sample_img, sample_lbl = val_subset[0]
    sample_input = sample_img.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(sample_input)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(logits.argmax(dim=-1).item())
        runner_up_idx = int(logits[0].argsort(descending=True)[1].item())

    print(f"  Item Ground Truth: '{FASHION_CLASSES[sample_lbl]}'")
    print(f"  Top Prediction:    '{FASHION_CLASSES[pred_idx]}' ({probs[0, pred_idx]:.1%})")
    print(f"  Runner-up Class:   '{FASHION_CLASSES[runner_up_idx]}' ({probs[0, runner_up_idx]:.1%})")

    # Target the last convolutional layer of ResNet-18 (layer4)
    target_conv = model.layer4[-1].conv2

    # A. Grad-CAM
    gradcam = GradCAM(model, target_layer=target_conv)
    gcam_map = gradcam.explain(sample_img, target_class=pred_idx)

    # Convert 1-channel normalized tensor to RGB uint8 image for display
    norm_img = sample_img * 0.3530 + 0.2860
    norm_img = norm_img.clamp(0, 1).repeat(3, 1, 1).permute(1, 2, 0).numpy()
    rgb_img = (norm_img * 255).astype("uint8")

    overlay = overlay_heatmap(rgb_img, gcam_map)
    os.makedirs("reports", exist_ok=True)
    cam_out = os.path.join("reports", "resnet_fashion_gradcam.png")
    Image.fromarray(overlay).save(cam_out)
    print(f"  [OK] Saved '{cam_out}' (Grad-CAM)")

    # B. HiResCAM (Elementwise gradients)
    hires = HiResCAM(model, target_layer=target_conv)
    hires_map = hires.explain(sample_img, target_class=pred_idx)
    print(f"  [OK] HiResCAM map computed (range: [{hires_map.min():.2f}, {hires_map.max():.2f}])")

    # C. Contrastive CAM (Why Top Class over Runner-up?)
    contrastive = ContrastiveCAM(model, target_layer=target_conv)
    contrast_map = contrastive.explain(sample_img, target_class=pred_idx, contrast_class=runner_up_idx)
    print(f"  [OK] Contrastive CAM: Visualizing discriminative features for '{FASHION_CLASSES[pred_idx]}' vs '{FASHION_CLASSES[runner_up_idx]}'")

    # 4. ModelDoctor Dataset-Wide Diagnostics
    print("\n[Step 4] Diagnosing Validation Set with ModelDoctor...")
    doctor = ModelDoctor(
        model,
        class_names=FASHION_CLASSES,
        normalize_mean=(0.2860, 0.2860, 0.2860),
        normalize_std=(0.3530, 0.3530, 0.3530),
    )
    report = doctor.diagnose(
        val_loader,
        n_clusters=4,
        n_example_images=8,
        cam_target_layer=target_conv,
        report_title="ResNet-18 Fashion-MNIST Diagnostic Report",
    )

    print("\n" + "=" * 50)
    print(report.summary_text())
    print("=" * 50)

    # Save interactive HTML dashboard
    os.makedirs("reports", exist_ok=True)
    report_html_path = os.path.join("reports", "resnet_fashion_diagnosis.html")
    report.save_html(report_html_path)
    print(f"[OK] Interactive HTML dashboard saved to '{report_html_path}'")

    # 5. One-Line Calibration Remediation
    print("\n[Step 5] Confidence Calibration Remediation...")
    scaler = TemperatureScaler(model)
    cal_summary = scaler.fit(val_loader)
    print(cal_summary.summary_text())

    calibrated_model = scaler.calibrated_model
    # Export metrics for CI/CD
    os.makedirs("reports", exist_ok=True)
    metrics_path = os.path.join("reports", "resnet_fashion_metrics.json")
    report.to_json(metrics_path)
    print(f"[OK] Exported '{metrics_path}'")

    print("\n" + "=" * 72)
    print("  EXPERIMENT COMPLETE!")
    print("  Artifacts generated:")
    print("    1. resnet_fashion_gradcam.png    (Visual overlay on real clothing image)")
    print("    2. resnet_fashion_diagnosis.html (Interactive HTML report with filter chips)")
    print("    3. resnet_fashion_metrics.json   (CI/CD metrics)")
    print("=" * 72)


if __name__ == "__main__":
    main()
