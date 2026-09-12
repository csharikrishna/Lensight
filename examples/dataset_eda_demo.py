"""
Demonstration: Plug-and-play Computer Vision Exploratory Data Analysis (DatasetExplorer).

Analyzes Fashion-MNIST for:
- Class representation & imbalance
- Exposure & contrast defects (underexposed, overexposed, low contrast)
- Sharpness & blur estimation
- Exact and near-duplicate detection
- Self-contained HTML report generation

Run with:
    python examples/dataset_eda_demo.py
"""

import sys
from pathlib import Path

# Ensure local lensight package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torchvision
import torchvision.transforms as transforms

from lensight import DatasetExplorer

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


def main():
    print("=" * 72)
    print("  LENSIGHT: PLUG-AND-PLAY CV EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 72)

    # 1. Load Dataset
    print("\n[Step 1] Loading Fashion-MNIST...")
    dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    print(f"[OK] Total dataset size: {len(dataset)} images")

    # 2. Run DatasetExplorer
    print("\n[Step 2] Profiling dataset with DatasetExplorer...")
    explorer = DatasetExplorer(dataset, class_names=FASHION_CLASSES)
    report = explorer.analyze(max_samples=1000)

    # 3. Print Summary
    print("\n" + report.summary_text())

    # 4. Save Interactive HTML Report
    os.makedirs("reports", exist_ok=True)
    html_out = os.path.join("reports", "fashion_eda_report.html")
    report.save_html(html_out)
    print(f"\n[OK] Interactive EDA Report written to '{html_out}'")
    print("  (Open in any browser to see the class distribution chart, health checks, and outlier gallery)")

    # 5. Export structured metrics
    metrics = report.to_dict()
    print("\n[Step 3] Sample Telemetry Dict:")
    print(f"  - Imbalance Ratio: {metrics['imbalance_ratio']:.2f}x")
    print(f"  - Mean Brightness: {metrics['exposure']['mean_brightness']:.1%}")
    print(f"  - Duplicate Pairs: {metrics['integrity']['duplicate_pairs_count']}")

    print("\n" + "=" * 72)
    print("  EDA AUDIT COMPLETE!")
    print("=" * 72)


if __name__ == "__main__":
    main()
