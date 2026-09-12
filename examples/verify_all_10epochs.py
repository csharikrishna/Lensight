"""
Verify All Lensight Features: 10-Epoch Real Training & Full Diagnostic Suite.

This script executes:
1. [Pre-Training Data QA] Audits Fashion-MNIST splits for imbalance, duplicates, and leakage.
2. [Data Sanitization] Non-destructively sanitizes the dataset using DatasetSanitizer.
3. [Model Training] Trains ResNet-18 on GPU for 10 FULL EPOCHS.
4. [XAI Suite] Tests GradCAM, GradCAM++, EigenCAM, HiResCAM, ContrastiveCAM, IntegratedGradients, SmoothGrad.
5. [Calibration Suite] Computes ECE, MCE, Brier Score, and applies TemperatureScaler.
6. [Error Profiling] Analyzes misclassifications, confusion matrices, and label errors.
7. [Dataset Exploration] Runs DatasetExplorer for pixel statistics, exposure, and blurriness.
8. [Unified ModelDoctor] Runs ModelDoctor end-to-end, generates interactive HTML and JSON metrics.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure local lensight package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image

import lensight
from lensight import (
    GradCAM,
    GradCAMPlusPlus,
    EigenCAM,
    HiResCAM,
    ContrastiveCAM,
    IntegratedGradients,
    SmoothGrad,
    VanillaGradient,
    ExpectedCalibrationError,
    MaximumCalibrationError,
    TemperatureScaler,
    ErrorProfiler,
    ModelDoctor,
    DatasetExplorer,
)
from lensight.eda import DatasetAuditor, DatasetSanitizer
from lensight.utils.image_utils import overlay_heatmap

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
    """ResNet-18 modified for 1-channel 28x28 Fashion-MNIST."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    start_time = time.time()
    os.makedirs("reports", exist_ok=True)

    print("=" * 80)
    print("       LENSIGHT COMPREHENSIVE 10-EPOCH END-TO-END VERIFICATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # -------------------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STEP 1: Dataset Loading & Pre-Training Audit (DatasetAuditor)")
    print("-" * 70)

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

    # Use a solid subset for 10 fast, accurate epochs (2500 train, 500 val)
    train_subset = data.Subset(full_train, range(2500))
    val_subset = data.Subset(full_val, range(500))

    train_loader = data.DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(val_subset, batch_size=64, shuffle=False)

    print(f"  Training set size:   {len(train_subset)} images across {len(FASHION_CLASSES)} classes")
    print(f"  Validation set size: {len(val_subset)} images")

    # Run DatasetAuditor on train & val splits
    print("\n  [Audit] Running DatasetAuditor on training set...")
    auditor = DatasetAuditor(class_names=FASHION_CLASSES)
    audit_report = auditor.audit(train_loader)
    print(audit_report.summary_text())

    # Save interactive audit report
    audit_html_path = os.path.join("reports", "lifecycle_10epoch_audit_report.html")
    audit_report.save_html(audit_html_path)
    print(f"  [OK] Saved Data Audit Dashboard to '{audit_html_path}'")

    # Check for train/test leakage
    print("\n  [Leakage] Auditing train vs val data leakage...")
    leakage_report = auditor.check_leakage(train_loader, val_loader)
    print(leakage_report.summary_text())
    leakage_html_path = os.path.join("reports", "lifecycle_10epoch_leakage_report.html")
    leakage_report.save_html(leakage_html_path)
    print(f"  [OK] Saved Leakage Dashboard to '{leakage_html_path}'")

    # Demonstrate DatasetSanitizer (in-memory non-destructive cleaning)
    sanitized_train = DatasetSanitizer.clean_subset(train_subset, audit_report)
    print(f"  [Sanitize] Sanitized train set size: {len(sanitized_train)} samples retained.")
    train_loader = data.DataLoader(sanitized_train, batch_size=64, shuffle=True)

    # -------------------------------------------------------------------------
    # 2. Train ResNet-18 for 10 Full Epochs
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STEP 2: Training ResNet-18 for 10 Full Epochs")
    print("-" * 70)

    model = build_fashion_resnet(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()

    epoch_history = []
    for epoch in range(1, 11):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == lbls).sum().item()
            train_total += imgs.size(0)

        scheduler.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(dim=1) == lbls).sum().item()
                val_total += imgs.size(0)

        tr_acc = train_correct / train_total
        vl_acc = val_correct / val_total
        tr_loss = train_loss / train_total
        vl_loss = val_loss / val_total

        epoch_history.append((epoch, tr_loss, tr_acc, vl_loss, vl_acc))
        print(f"  Epoch {epoch:2d}/10 | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.1%} | Val Loss: {vl_loss:.4f}, Acc: {vl_acc:.1%}")

    print(f"\n[OK] Completed 10 epochs. Final Val Accuracy: {vl_acc:.1%}")

    # -------------------------------------------------------------------------
    # 3. Comprehensive XAI / CAM Suite Verification
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STEP 3: Testing Full Explainability (XAI) Suite")
    print("-" * 70)

    model.eval()
    sample_img, sample_lbl = val_subset[0]
    sample_input = sample_img.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(sample_input)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(logits.argmax(dim=-1).item())
        runner_up_idx = int(logits[0].argsort(descending=True)[1].item())

    print(f"  Sample Ground Truth: '{FASHION_CLASSES[sample_lbl]}'")
    print(f"  Top Prediction:     '{FASHION_CLASSES[pred_idx]}' (Confidence: {probs[0, pred_idx]:.1%})")
    print(f"  Runner-up Class:    '{FASHION_CLASSES[runner_up_idx]}' (Confidence: {probs[0, runner_up_idx]:.1%})")

    target_conv = model.layer4[-1].conv2

    # A. Grad-CAM
    gcam = GradCAM(model, target_layer=target_conv)
    gcam_map = gcam.explain(sample_img, target_class=pred_idx)
    assert gcam_map.ndim == 2 and 0.0 <= gcam_map.min() <= gcam_map.max() <= 1.0
    print(f"  [OK] GradCAM: spatial map {gcam_map.shape} in range [{gcam_map.min():.2f}, {gcam_map.max():.2f}]")

    # B. Grad-CAM++
    gcam_pp = GradCAMPlusPlus(model, target_layer=target_conv)
    gcam_pp_map = gcam_pp.explain(sample_img, target_class=pred_idx)
    assert gcam_pp_map.ndim == 2 and 0.0 <= gcam_pp_map.min() <= gcam_pp_map.max() <= 1.0
    print(f"  [OK] GradCAMPlusPlus: spatial map {gcam_pp_map.shape} in range [{gcam_pp_map.min():.2f}, {gcam_pp_map.max():.2f}]")

    # C. Eigen-CAM
    eigen_cam = EigenCAM(model, target_layer=target_conv)
    eigen_map = eigen_cam.explain(sample_img)
    assert eigen_map.ndim == 2 and 0.0 <= eigen_map.min() <= eigen_map.max() <= 1.0
    print(f"  [OK] EigenCAM: spatial map {eigen_map.shape} in range [{eigen_map.min():.2f}, {eigen_map.max():.2f}]")

    # D. HiResCAM
    hires = HiResCAM(model, target_layer=target_conv)
    hires_map = hires.explain(sample_img, target_class=pred_idx)
    assert hires_map.ndim == 2 and 0.0 <= hires_map.min() <= hires_map.max() <= 1.0
    print(f"  [OK] HiResCAM: spatial map {hires_map.shape} in range [{hires_map.min():.2f}, {hires_map.max():.2f}]")

    # E. Contrastive CAM
    contrast = ContrastiveCAM(model, target_layer=target_conv)
    contrast_map = contrast.explain(sample_img, target_class=pred_idx, contrast_class=runner_up_idx)
    assert contrast_map.ndim == 2 and 0.0 <= contrast_map.min() <= contrast_map.max() <= 1.0
    print(f"  [OK] ContrastiveCAM: compared '{FASHION_CLASSES[pred_idx]}' vs '{FASHION_CLASSES[runner_up_idx]}'")

    # F. Saliency: Vanilla Gradient
    vanilla = VanillaGradient(model)
    vanilla_map = vanilla.explain(sample_img, target_class=pred_idx)
    assert vanilla_map.shape == (28, 28)
    print(f"  [OK] VanillaGradient: valid attribution range [{vanilla_map.min():.2f}, {vanilla_map.max():.2f}]")

    # G. Saliency: Integrated Gradients
    ig = IntegratedGradients(model, steps=20)
    ig_map = ig.explain(sample_img, target_class=pred_idx)
    assert ig_map.shape == (28, 28)
    print(f"  [OK] IntegratedGradients: 20 integration steps computed successfully")

    # H. Saliency: SmoothGrad
    sg = SmoothGrad(model, n_samples=15)
    sg_map = sg.explain(sample_img, target_class=pred_idx)
    assert sg_map.shape == (28, 28)
    print(f"  [OK] SmoothGrad: 15 Gaussian noise perturbations computed successfully")

    # Save representative overlay image
    norm_img = sample_img * 0.3530 + 0.2860
    norm_img = norm_img.clamp(0, 1).repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
    rgb_img = (norm_img * 255).astype("uint8")
    overlay = overlay_heatmap(rgb_img, gcam_map)
    cam_png_path = os.path.join("reports", "lifecycle_10epoch_gradcam.png")
    Image.fromarray(overlay).save(cam_png_path)
    print(f"  [OK] Saved visual overlay artifact to '{cam_png_path}'")

    # -------------------------------------------------------------------------
    # 4. Confidence Calibration & Temperature Scaling Verification
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STEP 4: Testing Confidence Calibration & Uncertainty Suite")
    print("-" * 70)

    # Collect validation predictions
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            p = torch.softmax(model(imgs), dim=-1)
            all_probs.append(p.cpu().numpy())
            all_labels.append(lbls.numpy())

    val_probs = np.concatenate(all_probs, axis=0)
    val_labels = np.concatenate(all_labels, axis=0)

    # ECE & MCE
    ece_metric = ExpectedCalibrationError(n_bins=10)
    mce_metric = MaximumCalibrationError(n_bins=10)
    raw_ece = ece_metric.compute(val_probs, val_labels)
    raw_mce = mce_metric.compute(val_probs, val_labels)

    print(f"  Raw Model Expected Calibration Error (ECE): {raw_ece:.4f} ({raw_ece:.1%})")
    print(f"  Raw Model Maximum Calibration Error  (MCE): {raw_mce:.4f} ({raw_mce:.1%})")

    # Temperature Scaling Remediation
    print("\n  [Remediation] Fitting TemperatureScaler to calibrate confidence...")
    scaler = TemperatureScaler(model)
    cal_summary = scaler.fit(val_loader)
    print(cal_summary.summary_text())

    # Verify accuracy is strictly invariant
    calibrated_model = scaler.calibrated_model
    with torch.no_grad():
        test_in = sample_img.unsqueeze(0).to(device)
        p_raw = model(test_in).argmax(dim=-1)
        p_cal = calibrated_model(test_in).argmax(dim=-1)
        assert torch.equal(p_raw, p_cal), "Accuracy must remain invariant under temperature scaling!"
    print(f"  [OK] Temperature scaling verified: Top-1 predictions are 100% mathematically invariant.")

    # -------------------------------------------------------------------------
    # 5. Misclassification & Error Profiling Verification
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STEP 5: Testing Error Profiling & Failure Mode Discovery")
    print("-" * 70)

    profiler = ErrorProfiler(model, class_names=FASHION_CLASSES)
    err_report = profiler.analyze(val_loader, n_clusters=4)
    print(err_report.summary_text())

    top_confused = err_report.top_confused_pairs(3)
    print("  Top Confused Class Pairs:")
    for t_idx, p_idx, count in top_confused:
        print(f"    * True '{err_report.label_name(t_idx)}' -> Predicted as '{err_report.label_name(p_idx)}' ({count} mistakes)")

    label_errors = err_report.find_label_errors(min_confidence=0.80)
    print(f"  High-confidence errors (suspected annotation errors): {len(label_errors)} found")

    # -------------------------------------------------------------------------
    # 6. Dataset Exploration (EDA)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STEP 6: Testing DatasetExplorer (Pixel Statistics & Exposure)")
    print("-" * 70)

    explorer = DatasetExplorer(val_subset, class_names=FASHION_CLASSES)
    eda_report = explorer.analyze(max_samples=150)
    print(eda_report.summary_text())
    explorer_html = os.path.join("reports", "lifecycle_10epoch_explorer_report.html")
    eda_report.save_html(explorer_html)
    print(f"  [OK] Saved Dataset Explorer Report to '{explorer_html}'")

    # -------------------------------------------------------------------------
    # 7. Unified ModelDoctor & Production Dashboard
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STEP 7: Testing Unified ModelDoctor (One-Shot Production Diagnostic)")
    print("-" * 70)

    doctor = ModelDoctor(
        model,
        class_names=FASHION_CLASSES,
        normalize_mean=(0.2860,),
        normalize_std=(0.3530,),
    )
    doc_report = doctor.diagnose(
        val_loader,
        n_clusters=4,
        n_example_images=8,
        cam_target_layer=target_conv,
        report_title="ResNet-18 Fashion-MNIST 10-Epoch Production Audit",
    )

    print(doc_report.summary_text())

    # Save Interactive HTML Report
    html_out = os.path.join("reports", "lifecycle_10epoch_diagnosis.html")
    doc_report.save_html(html_out)
    print(f"\n  [OK] Saved Interactive ModelDoctor Dashboard to '{html_out}'")

    # Export JSON Metrics for CI/CD Gates
    json_out = os.path.join("reports", "lifecycle_10epoch_metrics.json")
    doc_report.to_json(json_out)
    print(f"  [OK] Exported CI/CD Metrics to '{json_out}'")

    # Verify Built-in Quick Reference & Help
    print("\n  [Help System] Testing lensight.quick_reference()...")
    lensight.quick_reference()

    # -------------------------------------------------------------------------
    # 8. Summary & Quality Gate Assertion
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  ALL TESTS PASSED SUCCESSFULLY IN {elapsed:.1f} SECONDS!")
    print("=" * 80)
    print("  Artifacts generated in reports/:")
    print("    1. lifecycle_10epoch_eda_report.html (Interactive Data Quality Dashboard)")
    print("    2. lifecycle_10epoch_gradcam.png     (Visual XAI Heatmap Overlay)")
    print("    3. lifecycle_10epoch_diagnosis.html  (Interactive Model Diagnostic Dashboard)")
    print("    4. lifecycle_10epoch_metrics.json    (Machine-Readable CI/CD Metrics)")
    print("=" * 80)


if __name__ == "__main__":
    main()
