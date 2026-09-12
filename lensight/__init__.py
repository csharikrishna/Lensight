"""
lensight
========

A diagnostic toolkit for debugging and interpreting PyTorch computer-vision
models: visual explanations (Grad-CAM family, saliency maps), failure-mode
analysis (misclassification clustering, confidence calibration), and an
HTML report generator that ties it all together.

Quickstart
----------
    from lensight import GradCAM, IntegratedGradients, ModelDoctor

    cam = GradCAM(model)
    heatmap = cam.explain(image_tensor, target_class=3)

    doctor = ModelDoctor(model, class_names=classes)
    report = doctor.diagnose(dataloader)
    report.save_html("diagnosis.html")
"""

from .cam.gradcam import GradCAM
from .cam.gradcam_plus_plus import GradCAMPlusPlus
from .cam.eigencam import EigenCAM
from .cam.hirescam import HiResCAM
from .cam.contrastive_cam import ContrastiveCAM
from .saliency.vanilla_grad import VanillaGradient
from .saliency.integrated_gradients import IntegratedGradients
from .saliency.smoothgrad import SmoothGrad
from .analysis.misclassification import (
    MisclassificationAnalyzer,
    MisclassificationReport,
    MisclassifiedSample,
)
from .analysis.confidence import (
    ConfidenceAnalyzer,
    CalibrationReport,
    ExpectedCalibrationError,
    MaximumCalibrationError,
)
from .analysis.temperature_scaling import TemperatureScaler, CalibratedModel

# Intuitive user-facing aliases
ErrorProfiler = MisclassificationAnalyzer

from .utils.model_utils import create_vit_reshape_transform
from .eda import (
    DatasetExplorer,
    EDAReport,
    DatasetAuditor,
    AuditReport,
    LeakageReport,
    CrossLabelReport,
    DatasetSanitizer,
    audit_dataset,
    check_leakage,
    check_cross_label,
)
from .doctor import ModelDoctor

__version__ = "0.2.0"


def help() -> None:
    """Print an interactive quick reference cheat-sheet for Lensight workflows."""
    guide = """
================================================================================
                          LENSIGHT QUICK REFERENCE
         Unified PyTorch Computer Vision Diagnostics & Data Health
================================================================================

[1] Visual Explanations (CAM & Saliency)
    from lensight import GradCAM, HiResCAM, ContrastiveCAM, IntegratedGradients

    cam = GradCAM(model)                      # Auto-detects target layer
    heatmap = cam.explain(image_tensor)       # Top predicted class
    hires_map = HiResCAM(model).explain(image_tensor)
    contrast_map = ContrastiveCAM(model).explain(image_tensor, target_class=0, contrast_class=1)

[2] End-to-End Model Failure Diagnosis & Quality Alerts
    from lensight import ModelDoctor

    doctor = ModelDoctor(model, class_names=CLASS_NAMES)
    report = doctor.diagnose(val_loader, n_clusters=4, include_eda=True)
    print(report.summary_text())
    report.save_html("reports/diagnosis.html") # Self-contained interactive dashboard

[3] Confidence Calibration & One-Line Temperature Scaling
    from lensight import TemperatureScaler

    scaler = TemperatureScaler(model)
    summary = scaler.fit(val_loader)          # Optimizes temperature T via NLL
    print(f"Calibrated ECE: {summary.calibrated_ece:.4f}")
    calibrated_model = scaler.calibrated_model # Deploys as drop-in nn.Module

[4] Pre-Training Dataset Health & Exploratory Data Analysis (EDA)
    from lensight import DatasetExplorer

    explorer = DatasetExplorer(class_names=CLASS_NAMES)
    eda_report = explorer.analyze(loader)     # Checks imbalance, blur & exposure
    eda_report.save_html("reports/eda.html")

[5] Deep Dataset Auditing, Leakage Detection & In-Memory Sanitization
    from lensight import DatasetAuditor, check_leakage, DatasetSanitizer

    # A. Audit duplicates & cross-label conflicts
    auditor = DatasetAuditor(class_names=CLASS_NAMES)
    audit_report = auditor.audit(raw_dataset, threshold=0)

    # B. Check train/test split leakage
    leak_report = check_leakage(train_loader, test_loader, threshold=0)

    # C. In-Memory Sanitization (no files touched on disk)
    clean_dataset = DatasetSanitizer.clean_subset(raw_dataset, audit_report)

[6] Command Line Interface (CLI)
    lensight audit --dataset <path> [--report reports/audit.html]
    lensight leakage --train <train_path> --test <test_path>
    lensight clean --dataset <path> --output <clean_path>
    lensight help
================================================================================
"""
    print(guide.strip())


quick_reference = help

__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "EigenCAM",
    "HiResCAM",
    "ContrastiveCAM",
    "VanillaGradient",
    "IntegratedGradients",
    "SmoothGrad",
    "MisclassificationAnalyzer",
    "ConfidenceAnalyzer",
    "TemperatureScaler",
    "CalibratedModel",
    "create_vit_reshape_transform",
    "DatasetExplorer",
    "EDAReport",
    "DatasetAuditor",
    "AuditReport",
    "LeakageReport",
    "CrossLabelReport",
    "DatasetSanitizer",
    "audit_dataset",
    "check_leakage",
    "check_cross_label",
    "ModelDoctor",
    "help",
    "quick_reference",
    "__version__",
]
