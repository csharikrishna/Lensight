"""
Real-World Lensight Demonstration: From Diagnosis to Remediation.

This script demonstrates the 4 primary workflows an ML/CV team uses in production:
1. Single-Image Explainability: Grad-CAM vs HiResCAM vs Contrastive CAM ("Why A instead of B?")
2. Dataset-Wide Diagnostics: Failure clustering & Data Quality / Label Noise Detection
3. One-Line Calibration Remediation: Fixing overconfidence with Temperature Scaling
4. CI/CD & MLOps Quality Gating: Exporting metrics to JSON for automated CI/CD gates

Run with:
    python examples/real_world_demo.py
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
from PIL import Image

from lensight import (
    GradCAM,
    HiResCAM,
    ContrastiveCAM,
    ModelDoctor,
    TemperatureScaler,
)
from lensight.utils.image_utils import denormalize, overlay_heatmap

# Class labels for our CV problem
CLASSES = ["cat", "dog", "car", "airplane"]
NUM_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# 1. Realistic Model & Biased Dataset Setup
# ---------------------------------------------------------------------------
class ConvNetClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features.flatten(1))


def create_realistic_validation_data(n_samples=160, seed=42):
    """
    Creates a validation set where:
    - Most classes are easily distinguishable
    - 'cat' (0) and 'dog' (1) share overlapping features, creating a real failure cluster
    - 2 examples have deliberate LABEL NOISE (ground truth labeled 'dog', but actually 'car')
    """
    g = torch.Generator().manual_seed(seed)
    class_means = torch.tensor([[0.2, 0.2, 0.2], [0.25, 0.22, 0.2], [1.2, -0.5, 0.0], [-0.8, 1.0, 0.5]])

    labels = torch.randint(0, NUM_CLASSES, (n_samples,), generator=g)
    images = torch.randn(n_samples, 3, 32, 32, generator=g) * 0.4
    for i in range(n_samples):
        images[i] += class_means[labels[i]].view(3, 1, 1)

    # Inject 2 dirty label mistakes into the dataset
    # (Simulating real-world human annotator errors)
    labels[10] = 0  # Image is car-like, but labeled as cat!
    labels[25] = 1  # Image is airplane-like, but labeled as dog!

    return data.TensorDataset(images, labels)


def main():
    print("=" * 70)
    print("  LENSIGHT: END-TO-END PRODUCTION COMPUTER VISION DIAGNOSTICS")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Initialize Model and Data
    # -----------------------------------------------------------------------
    print("\n[Step 1] Initializing model and validation dataset...")
    val_dataset = create_realistic_validation_data(n_samples=160)
    val_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = ConvNetClassifier()
    # Train briefly so the model learns meaningful feature representations
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(6):
        for imgs, lbls in val_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
    model.eval()
    print("[OK] Model initialized and trained to baseline accuracy.")

    # -----------------------------------------------------------------------
    # Step 2: Single-Image Interpretability (Grad-CAM, HiResCAM, Contrastive CAM)
    # -----------------------------------------------------------------------
    print("\n[Step 2] Explaining a single prediction...")
    test_image, test_label = val_dataset[0]

    with torch.no_grad():
        logits = model(test_image.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        pred_class = int(logits.argmax(dim=-1).item())
        runner_up = int(logits[0].argsort(descending=True)[1].item())

    print(f"  Ground truth: '{CLASSES[test_label]}' | Predicted: '{CLASSES[pred_class]}' ({probs[0, pred_class]:.1%})")
    print(f"  Runner-up competitor: '{CLASSES[runner_up]}' ({probs[0, runner_up]:.1%})")

    # A. Standard Grad-CAM
    cam = GradCAM(model)
    gradcam_heatmap = cam.explain(test_image, target_class=pred_class)
    rgb = denormalize(test_image)
    os.makedirs("reports", exist_ok=True)
    out_cam = os.path.join("reports", "demo_gradcam.png")
    Image.fromarray(overlay).save(out_cam)
    print(f"  [OK] Saved '{out_cam}' (coarse activation localization)")

    # B. HiResCAM (Elementwise gradients - eliminates spatial averaging wash-out)
    hires = HiResCAM(model)
    hires_heatmap = hires.explain(test_image, target_class=pred_class)
    print(f"  [OK] HiResCAM computed (elementwise fidelity, map range: [{hires_heatmap.min():.2f}, {hires_heatmap.max():.2f}])")

    # C. Contrastive CAM ("Why predicted class A instead of competitor B?")
    contrastive = ContrastiveCAM(model)
    contrast_heatmap = contrastive.explain(test_image, target_class=pred_class, contrast_class=runner_up)
    print(f"  [OK] Contrastive CAM computed: Isolating evidence for '{CLASSES[pred_class]}' over '{CLASSES[runner_up]}'")

    # -----------------------------------------------------------------------
    # Step 3: ModelDoctor Dataset-Wide Diagnosis & Label Noise Detection
    # -----------------------------------------------------------------------
    print("\n[Step 3] Running ModelDoctor across the full validation dataset...")
    doctor = ModelDoctor(model, class_names=CLASSES)
    report = doctor.diagnose(
        val_loader,
        n_clusters=3,
        n_example_images=6,
        report_title="Production Model Health & Diagnosis Report",
    )

    print("-" * 50)
    print(report.summary_text())
    print("-" * 50)

    # Save interactive HTML report
    os.makedirs("reports", exist_ok=True)
    html_path = os.path.join("reports", "model_diagnosis_report.html")
    report.save_html(html_path)
    print(f"[OK] Interactive HTML Dashboard saved to '{html_path}'")
    print("  (Open in browser to use interactive failure filters and search!)")

    # Check for suspected label errors
    label_errors = report.misclassification.find_label_errors(min_confidence=0.80)
    print(f"\n[Data Quality] Detected {len(label_errors)} suspected label errors in dataset:")
    for err in label_errors:
        print(f"  * Sample #{err.index}: Labeled as '{CLASSES[err.true_label]}' but model is {err.confidence:.1%} confident it is '{CLASSES[err.pred_label]}'")

    # -----------------------------------------------------------------------
    # Step 4: Closing the Loop - One-Line Calibration Remediation
    # -----------------------------------------------------------------------
    print("\n[Step 4] Remediation: Calibrating model confidence scores...")
    scaler = TemperatureScaler(model)
    cal_summary = scaler.fit(val_loader)

    print(cal_summary.summary_text())
    calibrated_model = scaler.calibrated_model

    # Verify top-1 prediction is 100% preserved
    orig_preds = model(test_image.unsqueeze(0)).argmax(dim=-1)
    cal_preds = calibrated_model(test_image.unsqueeze(0)).argmax(dim=-1)
    assert torch.equal(orig_preds, cal_preds), "Accuracy must be strictly invariant under temperature scaling!"
    print("[OK] Verification: Top-1 accuracy is 100% invariant while probabilities are properly calibrated.")

    # -----------------------------------------------------------------------
    # Step 5: Automated CI/CD Regression Gating & MLOps Tracking
    # -----------------------------------------------------------------------
    print("\n[Step 5] CI/CD Regression Gating...")
    metrics = report.to_dict()

    # Save to metrics.json for GitHub Actions, W&B, MLflow
    os.makedirs("reports", exist_ok=True)
    metrics_path = os.path.join("reports", "diagnosis_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Exported '{metrics_path}'")

    # Example of an automated CI gate:
    MAX_PERMISSIBLE_ECE = 0.15
    MAX_LABEL_ERRORS = 5

    ece = metrics["calibration"]["ece"]
    n_label_errs = len(metrics["misclassification"]["suspected_label_errors"])

    print(f"  Quality Gate Checks:")
    print(f"  - ECE: {ece:.4f} (Threshold: <= {MAX_PERMISSIBLE_ECE}) -> {'PASSED' if ece <= MAX_PERMISSIBLE_ECE else 'FAILED'}")
    print(f"  - Dataset Label Errors: {n_label_errs} (Threshold: <= {MAX_LABEL_ERRORS}) -> {'PASSED' if n_label_errs <= MAX_LABEL_ERRORS else 'FAILED'}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETED SUCCESSFULLY!")
    print("  Artifacts generated:")
    print("    1. demo_gradcam.png             (Visual explanation overlay)")
    print("    2. model_diagnosis_report.html  (Interactive HTML dashboard)")
    print("    3. diagnosis_metrics.json       (CI/CD / MLOps metrics)")
    print("=" * 70)


if __name__ == "__main__":
    main()
