# Lensight Real-World Use Cases & Engineering Guide

This guide documents the core real-world problems `lensight` solves in computer vision engineering, complete with code examples, actual benchmark outputs from a **ResNet-18 model trained on Fashion-MNIST**, and actionable playbooks.

---

## Table of Contents

1. [Use Case 1: Explaining Single Predictions ("Why Did It Predict That?")](#use-case-1-explaining-single-predictions)
2. [Use Case 2: Dataset-Wide Failure Auditing & Cluster Discovery](#use-case-2-dataset-wide-failure-auditing)
3. [Use Case 3: Data Quality & Annotation Noise Detection](#use-case-3-data-quality--annotation-noise-detection)
4. [Use Case 4: Fixing Overconfidence with Temperature Scaling](#use-case-4-fixing-overconfidence-with-temperature-scaling)
5. [Use Case 5: Vision Transformer (ViT) Attribution](#use-case-5-vision-transformer-vit-attribution)
6. [Use Case 6: Automated CI/CD Quality Gates & MLOps Tracking](#use-case-6-automated-cicd-quality-gates)
7. [Use Case 7: Pre-Training Data Sanity Checks & Exploratory Data Analysis (EDA)](#use-case-7-pre-training-data-sanity-checks--exploratory-data-analysis-eda)
8. [Use Case 8: Train/Test Leakage Detection, Cross-Label Auditing & Dataset Sanitization](#use-case-8-traintest-leakage-detection-cross-label-auditing--dataset-sanitization)
9. [Reference Benchmark Run (ResNet-18 on Fashion-MNIST)](#reference-benchmark-run)

---

## Use Case 1: Explaining Single Predictions

### The Problem
During QA or customer audits, a classifier predicts class A, but stakeholders disagree or find it suspicious. Traditional models provide only a confidence scalar (e.g. "98.5% confident"), giving no indication of *why* the model made that decision or if it is relying on spurious background artifacts.

### Toolkit Solution
`lensight` offers a trio of visual explainers:
1. **Grad-CAM**: Fast, classic activation localization.
2. **HiResCAM**: Eliminates spatial pooling before summation for high-resolution, pixel-precise localization (NeurIPS 2020).
3. **Contrastive CAM**: Answers *"Why class A instead of competitor class B?"* by backpropagating the logit difference $z_{\text{target}} - z_{\text{contrast}}$.

### Code Example
```python
import torch
from PIL import Image
from lensight import GradCAM, HiResCAM, ContrastiveCAM
from lensight.utils.image_utils import denormalize, overlay_heatmap

model.eval()

# 1. Standard Grad-CAM
cam = GradCAM(model, target_layer=model.layer4[-1].conv2)
heatmap = cam.explain(image_tensor, target_class=pred_idx)
rgb = denormalize(image_tensor, mean=(0.286,), std=(0.353,))
overlay = overlay_heatmap(rgb, heatmap)
Image.fromarray(overlay).save("gradcam_explanation.png")

# 2. HiResCAM (Elementwise gradient fidelity)
hires = HiResCAM(model, target_layer=model.layer4[-1].conv2)
hires_map = hires.explain(image_tensor, target_class=pred_idx)

# 3. Contrastive CAM (Why 'Ankle boot' instead of runner-up 'Sneaker'?)
contrastive = ContrastiveCAM(model, target_layer=model.layer4[-1].conv2)
contrast_map = contrastive.explain(
    image_tensor, 
    target_class=ankle_boot_idx, 
    contrast_class=sneaker_idx
)
```

### Interpretation
* **Grad-CAM** highlights the general region (e.g. the entire shoe).
* **HiResCAM** sharply localizes the shaft and heel contour without blurring.
* **Contrastive CAM** highlights specifically the high-top collar that separates an ankle boot from a low-cut sneaker.

---

## Use Case 2: Dataset-Wide Failure Auditing

### The Problem
When evaluating a model on 10,000 validation images with 80% accuracy, teams are left with a flat list of 2,000 misclassified images. Browsing them randomly hides systemic patterns:
* Is the model failing on specific lighting conditions?
* Is it confusing a specific category pair (e.g. Coat vs Pullover)?
* Are the failures random noise or structural blind spots?

### Toolkit Solution
`ModelDoctor` automates the entire validation audit:
1. Extracts penultimate-layer representations of every error.
2. Clusters errors using k-means into **semantic failure clusters**.
3. Compiles an **interactive HTML dashboard** with zero external dependencies.

### Code Example
```python
from lensight import ModelDoctor

doctor = ModelDoctor(model, class_names=CLASS_NAMES)
report = doctor.diagnose(
    val_dataloader,
    n_clusters=4,
    n_example_images=8,
    report_title="ResNet-18 Validation Diagnosis",
)

# Print terminal summary
print(report.summary_text())

# Save interactive dashboard
report.save_html("diagnosis_report.html")

# In Jupyter / Colab notebooks, render inline:
report
```

### Real Benchmark Output
```
Accuracy: 79.33% (238/300)
Errors analyzed: 62
Most confused pairs (true -> predicted : count):
  Coat -> Pullover : 12
  Shirt -> Pullover : 11
  Coat -> Dress : 6
  Shirt -> T-shirt/top : 5
  Sandal -> Sneaker : 5
Failure clusters found: 4
  cluster 3 (n=19): mostly Coat -> Dress, avg confidence 0.63
  cluster 1 (n=17): mostly Shirt -> Pullover, avg confidence 0.64
  cluster 2 (n=17): mostly Coat -> Pullover, avg confidence 0.73
  cluster 0 (n=9): mostly Sandal -> Sneaker, avg confidence 0.76
```

### Actionable Takeaways
* **Cluster 2 & 1:** The model struggles to distinguish long-sleeve outerwear (Coat, Shirt, Pullover). Solution: Collect more cropped collar and button annotations.
* **Cluster 0:** Summer footwear vs athletic sneakers confusion. Solution: Augment data with sole/strap contrast.

---

## Use Case 3: Data Quality & Annotation Noise Detection

### The Problem
In commercial computer vision, up to 10% of validation errors are not model flaws—they are **bad ground-truth annotations** created by human labelers. If developers blindly optimize against bad labels, the model learns corrupted patterns.

### Toolkit Solution
`lensight` identifies **high-confidence misclassifications**: examples where the model is $\ge 80\%$ confident in class $A$, but the dataset label states class $B$. In real datasets, these are primary candidates for label errors.

> **Methodology Note**: Unlike dedicated frameworks such as `cleanlab` (which estimate joint distributions via out-of-fold cross-validated confident learning), Lensight's `find_label_errors` is designed as a **fast, zero-overhead complementary triage heuristic**. It runs instantly during standard validation passes to surface high-priority suspected mislabels for manual inspection without requiring multi-fold retraining.

### Code Example
```python
# Extract suspected label noise
label_errors = report.misclassification.find_label_errors(min_confidence=0.80)

print(f"Found {len(label_errors)} annotation anomalies:")
for err in label_errors:
    print(
        f"  Sample #{err.index}: Dataset label is '{CLASSES[err.true_label]}', "
        f"but model is {err.confidence:.1%} confident it is '{CLASSES[err.pred_label]}'"
    )
```

### Real Benchmark Catch
```
Suspected label errors (conf >= 80%): 22
  idx 21: true 'Sandal' -> pred 'Sneaker' (99.6%)
  idx 68: true 'Ankle boot' -> pred 'Sneaker' (99.5%)
  idx 147: true 'Shirt' -> pred 'Dress' (97.1%)
```
Inspecting sample `#21` immediately reveals an athletic shoe with open mesh that was mislabeled as a sandal in the dataset.

---

## Use Case 4: Fixing Overconfidence with Temperature Scaling

### The Problem
Modern deep networks are notorious for **miscalibration**: saying "99% confident" on cases where empirical accuracy is only 75%. Downstream systems (e.g. auto-accept thresholds, active learning, human-in-the-loop review) fail when confidence cannot be trusted.

### Toolkit Solution
`TemperatureScaler` implements post-hoc temperature scaling (Guo et al., ICML 2017):
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
Optimizing scalar $T > 0$ via NLL loss aligns confidence with empirical accuracy while **guaranteeing 100% preservation of top-1 classification accuracy**.

### Code Example
```python
from lensight import TemperatureScaler

scaler = TemperatureScaler(model)
summary = scaler.fit(val_dataloader)

print(summary.summary_text())
# Output:
#   Temperature: 1.440
#   ECE: 0.0750 -> 0.0501 (-33.2% error)
#   MCE: 0.3047 -> 0.2599
#   NLL: 0.6699 -> 0.6308

# Deploy the calibrated model directly:
calibrated_model = scaler.calibrated_model
probs = calibrated_model.predict_proba(image_tensor)
```

---

## Use Case 5: Vision Transformer (ViT) Attribution

### The Problem
Vision Transformers (ViT, DeiT, Swin, DINO) output sequences of patch tokens $(B, N, D)$ rather than 2D convolutional feature maps $(B, C, H, W)$. Passing them to conventional CAM explainers causes shape mismatch crashes.

### Toolkit Solution
`lensight` includes `create_vit_reshape_transform` to convert transformer tokens back into 2D spatial maps excluding the `[CLS]` token.

### Code Example
```python
import torchvision.models as models
from lensight import GradCAM, create_vit_reshape_transform

vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).eval()

cam = GradCAM(
    vit,
    target_layer=vit.encoder.layers[-1].ln_1,
    reshape_transform=create_vit_reshape_transform(has_cls_token=True),
)

heatmap = cam.explain(image_tensor)
```

---

## Use Case 6: Automated CI/CD Quality Gates

### The Problem
Models retrained with new data often improve overall accuracy while silently regressing on calibration error, or introducing severe confusion on critical classes.

### Toolkit Solution
Export structured reports via `report.to_json()` or `report.to_dict()` and enforce quality gates in GitHub Actions or CI/CD pipelines.

### Code Example
```python
# In train_and_eval.py inside your CI runner:
report = doctor.diagnose(val_loader)
report.to_json("diagnosis_metrics.json")
metrics = report.to_dict()

# Automated CI Quality Gates:
assert metrics["misclassification"]["accuracy"] >= 0.78, "Accuracy regression!"
assert metrics["calibration"]["ece"] <= 0.08, "Model exceeds permissible calibration error!"
assert len(metrics["misclassification"]["suspected_label_errors"]) <= 25, "Dataset contamination exceeded limit!"
print("[OK] Model passed all production deployment criteria.")
```

---

## Use Case 7: Pre-Training Data Sanity Checks & Exploratory Data Analysis (EDA)

### The Problem
Engineers frequently jump straight to training models without first auditing raw image datasets. Generic tabular EDA tools (like pandas profiling) provide little value for vision datasets, leaving critical computer vision failure modes unnoticed until training fails or test performance collapses:
* **Severe Class Imbalance**: Hidden long-tail distributions where minority classes have <10% of majority samples, leading models to predict only majority classes.
* **Extreme Exposure Defects**: Images that are completely pitch-black (underexposed) or fully blown out/saturated (overexposed), starving convolutions of informative gradients.
* **Blurry & Out-of-Focus Samples**: Low-quality images degraded by sensor noise or motion blur that introduce confusion.
* **Exact & Perceptual Duplicates**: Identical or near-identical images present across training and validation splits (**data leakage**), causing artificially inflated validation metrics.
* **Inconsistent Image Dimensions**: Silent resizing artifacts or unexpected aspect ratio distortions.

### Toolkit Solution
`lensight.eda.DatasetExplorer` provides a lightweight, plug-and-play EDA solution tailored specifically to computer vision:
1. **Zero Bloat & Streaming**: Profiles datasets directly from PyTorch `DataLoader` or `Dataset` instances without buffering entire datasets in RAM.
2. **5 Core CV Health Checks**:
   * Class distribution and imbalance ratio ($N_{\text{max}} / N_{\text{min}}$).
   * Exposure and contrast outliers (mean intensity and standard deviation).
   * Laplacian filter blur detection (variance of $3 \times 3$ Laplacian operator).
   * 64-bit perceptual difference hashing (`dhash`) to detect duplicate and near-duplicate images.
   * Spatial resolution and aspect ratio consistency checks.
3. **Interactive Dark-Themed HTML Dashboard**: Generates standalone reports with interactive tabs, SVG charts, and base64 outlier galleries.
4. **Seamless `ModelDoctor` Integration**: Simply pass `include_eda=True` to `doctor.diagnose()` to audit data health and model performance in a single pass.

### Code Example 1: Standalone Dataset Health Check
```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lensight.eda import DatasetExplorer

# 1. Load your computer vision dataset
transform = transforms.ToTensor()
dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 2. Run plug-and-play EDA
explorer = DatasetExplorer(
    class_names=["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    max_samples=1000  # fast sampling or None for full dataset
)
report = explorer.analyze(loader)

# 3. Print terminal diagnostic summary
print(report.summary_text())

# 4. Save interactive standalone HTML report
report.save_html("fashion_eda_report.html")

# 5. Extract structured metrics for CI/CD gates
metrics = report.to_dict()
assert metrics["imbalance_ratio"] < 3.0, "Class imbalance exceeds tolerance!"
assert len(metrics["duplicate_pairs"]) == 0, "Duplicate images found in evaluation split!"
```

### Code Example 2: Unified Model & Data Audit with `ModelDoctor`
```python
from lensight import ModelDoctor

doctor = ModelDoctor(model, class_names=CLASS_NAMES)

# Perform model evaluation + dataset EDA simultaneously
report = doctor.diagnose(
    val_dataloader,
    include_eda=True,  # Enables embedded dataset health analysis
    report_title="ResNet-18 Production Health Audit"
)

# Access EDA findings directly
print(f"Data Health Warnings: {len(report.eda_report.warnings)}")
report.save_html("full_audit_report.html")
```

---

## Use Case 8: Train/Test Leakage Detection, Cross-Label Auditing & Dataset Sanitization

### The Problem
Real-world datasets suffer from silent dataset contamination bugs that severely degrade models:
1. **Train/Test Leakage**: Identical or near-identical images appear in both train and validation splits. When test images are leaked, evaluation accuracy is artificially inflated, giving teams false confidence before deployment.
2. **Cross-Label Contradictions**: The exact same image appears under two different class labels (e.g. an MRI scan labeled as both *glioma* and *meningioma*, or a shoe labeled as both *Coat* and *Dress*). This injects directly contradictory loss gradients during training.
3. **Transitive Redundancy Chains**: Near-duplicates form chains ($A \sim B$ and $B \sim C$). Naive pairwise filtering keeps multiple redundant copies depending on the order files were scanned.
4. **Destructive Scripts**: Legacy tools delete or copy files directly on disk, risking data loss and breaking reproducible PyTorch pipelines.

### Toolkit Solution
`lensight.eda.DatasetAuditor` and `DatasetSanitizer` provide a complete auditing and remediation workflow:
* **Vectorized Perceptual Hashing**: Fast pure NumPy 64-bit perceptual hashing (`dhash`, `ahash`, `phash`) with vectorized pairwise XOR matrix calculations ($12\times$ to $16\times$ faster than nested python loops).
* **Transitive Union-Find Clustering with Smart Survivor Selection**: Collapses chains of near-duplicates into clean clusters, automatically keeping the sample with the **highest Laplacian sharpness score** and resolution.
* **Train/Test Split Leakage Detection**: Cross-references validation/test images against training sets and provides per-class leakage percentages.
* **In-Memory PyTorch Sanitization**: `DatasetSanitizer.clean_subset` produces a clean `torch.utils.data.Subset` with zero disk modification.
* **Interactive Dark-Themed HTML Audit Dashboard**: Generates self-contained dashboards with interactive tabs for Duplicates, Cross-Label Conflicts, and Leakage.

### Code Example: Auditing & Sanitizing in PyTorch
```python
from lensight.eda import DatasetAuditor, check_leakage, DatasetSanitizer

# 1. Audit dataset for duplicates & cross-label contradictions
auditor = DatasetAuditor(class_names=CLASS_NAMES)
report = auditor.audit(raw_dataset, threshold=0)

# Print terminal diagnostic summary
print(report.summary_text())

# Export interactive visual HTML report
report.save_html("audit_report.html")

# 2. Check for Train/Test Leakage
leak_report = check_leakage(train_loader, val_loader, threshold=0)
print(f"Leaked validation images: {leak_report.leaked_count} ({leak_report.leakage_rate * 100:.2f}%)")

# 3. In-Memory Sanitization (zero disk mutation)
clean_dataset = DatasetSanitizer.clean_subset(raw_dataset, report)
print(f"Clean samples ready for training: {len(clean_dataset)}")

# 4. Safe Directory Export (if working with image folders)
DatasetSanitizer.export_clean_directory("data/raw_images", "data/clean_images", report, mode="copy")
```

### CLI Command Reference
```bash
# Audit an image dataset folder:
lensight audit --dataset data/dataset/ --report audit_report.html

# Check leakage between train and test splits:
lensight leakage --train data/train/ --test data/test/

# Copy only clean survivors to a new directory:
lensight clean --dataset data/dataset/ --output data/clean_dataset/
```

---

## Reference Benchmark Run

To replicate the complete workflows demonstrated across all use cases:

### 1. Model Training, Diagnosis & Visual Attribution
```bash
python examples/train_and_diagnose_resnet.py
```
**Artifacts Generated in `reports/`**:
* `reports/resnet_fashion_gradcam.png`: High-resolution visual explanation overlay on a real test garment.
* `reports/resnet_fashion_diagnosis.html`: Self-contained interactive dashboard with filter chips, search input, and SVG charts.
* `reports/resnet_fashion_metrics.json`: Structured diagnostic telemetry ready for MLOps ingestion.

### 2. Plug-and-Play Dataset Exploratory Data Analysis (EDA)
```bash
python examples/dataset_eda_demo.py
```
**Artifacts Generated in `reports/`**:
* `reports/fashion_eda_report.html`: Standalone interactive dataset health report with class balance charts, exposure metrics, blur analysis, duplicate pairs, and outlier image galleries.

### 3. Deep Dataset Auditing, Leakage Detection & Sanitization
```bash
python examples/dataset_audit_demo.py
```
**Artifacts Generated in `reports/`**:
* `reports/audit_report.html`: Interactive dark-themed audit dashboard with tabs for Duplicate Clusters (survivor vs duplicates), Cross-Label Contradictions, and Train/Test Leakage.
* `reports/dataset_audit_manifest.json`: Structured audit telemetry describing each image's status.
