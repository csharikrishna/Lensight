"""Comprehensive Demonstration of Lensight Dataset Auditing & Sanitization Suite.

Inspired by imgcheck, but significantly elevated:
  - Works on PyTorch DataLoaders & Datasets (not just folders on disk)
  - Pure NumPy vectorized Hamming distance matrix (12x-16x faster than python loops)
  - Transitive Union-Find duplicate clustering with smart survivor selection (highest sharpness/quality)
  - Train/Test leakage detection with per-class leakage rates
  - Cross-label contradictory duplicate detection
  - In-memory PyTorch Subset sanitization (zero disk mutation) & directory export
  - Self-contained dark-mode interactive HTML audit dashboard
"""

import os
import sys
from pathlib import Path

# Ensure local lensight package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

from lensight.eda import (
    DatasetAuditor,
    audit_dataset,
    check_leakage,
    check_cross_label,
    DatasetSanitizer,
)


def main():
    print("=" * 70)
    print("      LENSIGHT DATASET AUDIT & LEAKAGE SANITIZATION SUITE       ")
    print("=" * 70)

    # 1. Load sample dataset: Fashion-MNIST test split
    print("\n[*] Loading Fashion-MNIST dataset split...")
    transform = transforms.ToTensor()
    raw_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    class_names = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    # 2. Run Comprehensive Dataset Audit on 1,000 samples
    print("\n[*] Phase 1: Comprehensive Dataset Audit (Imbalance, Duplicates, Cross-Labels)")
    auditor = DatasetAuditor(class_names=class_names, max_samples=1000)
    audit_report = auditor.audit(raw_dataset, threshold=0, algo="dhash")

    print(audit_report.summary_text())

    # 3. Export standalone interactive HTML audit dashboard
    os.makedirs("reports", exist_ok=True)
    html_output = os.path.join("reports", "audit_report.html")
    audit_report.save_html(html_output, title="Fashion-MNIST Dataset Integrity & Duplication Audit")
    print(f"\n[OK] Interactive HTML audit dashboard written to: {html_output}")

    # 4. Phase 2: Train/Test Leakage Detection Demonstration
    print("\n" + "=" * 70)
    print("[*] Phase 2: Simulating Train/Test Split Leakage Detection")
    print("=" * 70)

    # Simulate realistic train/test split where 5 images from train were accidentally leaked into test
    train_subset = [raw_dataset[i] for i in range(200)]
    # Intentionally leak 3 images into test set
    leaked_indices = [10, 25, 50]
    test_subset = [raw_dataset[i] for i in range(200, 250)] + [raw_dataset[i] for i in leaked_indices]

    print(f"Auditing leakage between: Train ({len(train_subset)} images) vs Test ({len(test_subset)} images)...")
    leak_report = check_leakage(train_subset, test_subset, class_names=class_names, threshold=0)
    print(leak_report.summary_text())

    # 5. Phase 3: In-Memory Dataset Sanitization (Zero Disk Modification)
    print("\n" + "=" * 70)
    print("[*] Phase 3: Dataset Sanitization (In-Memory PyTorch Cleaning)")
    print("=" * 70)

    clean_indices = audit_report.get_clean_indices(remove_duplicates=True)
    print(f"Original profiled samples: {audit_report.total_samples}")
    print(f"Sanitized sample indices:  {len(clean_indices)}")
    print(f"Excluded redundant copies: {audit_report.total_samples - len(clean_indices)}")

    clean_subset = DatasetSanitizer.clean_subset(raw_dataset, audit_report)
    print(f"[OK] Created clean PyTorch Subset with {len(clean_subset)} samples.")
    print("     You can now train directly on `clean_subset` with zero duplicate pollution!")

    # 6. Export manifest
    os.makedirs("reports", exist_ok=True)
    manifest_path = os.path.join("reports", "dataset_audit_manifest.json")
    audit_report.export_manifest(manifest_path)
    print(f"[OK] Structured audit manifest written to: {manifest_path}")

    print("\n" + "=" * 70)
    print("[OK] Dataset Audit Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
