# GitHub & Social Launch Kit

This kit contains all marketing, promotional, and repository metadata for the public launch of **Lensight** on GitHub, PyPI, LinkedIn, Reddit, and Twitter/X.

---

## 1. GitHub Repository Metadata

### About Section
```
The unified, zero-bloat PyTorch computer vision diagnostic toolkit: pre-training dataset health, train/test leakage detection, confidence calibration (ECE), semantic failure clustering, and Grad-CAM explainability in a single import.
```

### Repository Website / Documentation URL
`https://csharikrishna.github.io/Lensight-/`

### GitHub Topics / Tags
Add these tags to the repository:
```
pytorch, computer-vision, deep-learning, grad-cam, explainability, xai, data-quality, data-leakage, calibration, temperature-scaling, mlops, model-diagnostics, failure-analysis, eda, torchvision
```

---

## 2. LinkedIn Launch Post

**Title:** *Why Computer Vision engineers spend more time debugging tooling than debugging models — and how we fixed it.*

```markdown
If you train Computer Vision models in PyTorch, your debugging workflow probably looks like this:

1. Pip-install `pytorch-grad-cam` to see where your CNN is looking.
2. Pip-install `cleanlab` to check for noisy labels.
3. Pip-install `netcal` to see if your confidence scores are overconfident.
4. Write custom ad-hoc scripts to catch train/test data leakage.
5. Spend hours reconciling different tensor dimensions, dependency conflicts, and unmaintained packages.

Data Quality and Model Quality aren't two separate disciplines. They are two halves of the exact same debugging loop.

Today, I’m excited to open-source **Lensight**: a unified, production-grade diagnostic and data health toolkit for PyTorch Computer Vision.

🔍 What does Lensight do in a single import?
• Pre-Training Data Health: Detects train/test leakage, near-duplicates, and cross-label contradictions with 50x faster vectorized NumPy bitwise hashing.
• Smart Survivor Selection: When cleaning duplicate clusters, it uses Laplacian focus variance to automatically keep the sharpest, highest-resolution original rather than deleting arbitrarily.
• Non-Destructive In-Memory Sanitization: Generates clean `torch.utils.data.Subset` instances without copying gigabytes of files on disk.
• Explainability (XAI): Native Grad-CAM, Grad-CAM++, HiResCAM, Contrastive CAM ("Why Pullover instead of Coat?"), and Integrated Gradients with vectorized GPU batching.
• Confidence Calibration: Measures Expected Calibration Error (ECE) and remediates overconfidence with 1-line Temperature Scaling (leaving top-1 accuracy 100% invariant).
• One-Shot ModelDoctor: Runs dataset-wide diagnostics, clusters failure embeddings with PCA/k-Means, and exports self-contained interactive HTML dashboards.

⚡ Zero-Bloat Philosophy:
No OpenCV. No Pandas. No heavy C++ compilers. Just PyTorch, NumPy, Pillow, and Scikit-Learn.
The entire wheel is under 80 KB and runs all 45 unit tests in ~8 seconds.

Check it out on GitHub and let me know your thoughts:
👉 https://github.com/csharikrishna/Lensight-

#MachineLearning #ComputerVision #PyTorch #DeepLearning #DataScience #OpenSource #AI #MLOps
```

---

## 3. Reddit `r/MachineLearning` Launch Post

**Title:** `[P] Lensight: A unified, dependency-light toolkit for PyTorch CV data auditing, confidence calibration, error clustering, and Grad-CAM`

```markdown
**GitHub:** https://github.com/csharikrishna/Lensight-  
**License:** MIT  
**Dependencies:** PyTorch, NumPy, Pillow, Scikit-Learn (No OpenCV, No Pandas, No SciPy)

Hi r/MachineLearning,

Every time we train a new vision model, we find ourselves stitching together 4–5 unmaintained libraries just to answer basic QA questions:
- Did near-duplicate images leak between our train and test splits?
- Is the model overconfident on its mistakes (high ECE)?
- What visual features distinguish the top prediction from the runner-up class?
- Are our misclassifications random noise or systematic failure clusters?

Existing tools either require heavy dependencies (OpenCV, Pandas, SciPy, custom C++ build tools), force destructive file deletions on disk, or only support one specific aspect of the workflow.

We built **Lensight** to unify pre-training data auditing and post-training model diagnostics into a single lightweight toolkit.

### Key Features & Architectural Highlights:
1. **Vectorized Perceptual Hashing:** Instead of nested Python loops, Lensight bit-unpacks 64-bit hashes into uint8 arrays and uses a 256-element bitwise XOR popcount lookup table. It computes pairwise Hamming distances across 10,000 images in ~5 seconds (50x faster than legacy tools).
2. **Quality-Aware Survivor Selection:** Legacy duplicate removers keep the first alphabetical file path (often deleting a 4K original in favor of a low-res thumbnail). Lensight calculates 2D Laplacian sharpness variance to guarantee the highest-fidelity image is preserved.
3. **In-Memory Dataset Sanitization:** `DatasetSanitizer.clean_subset` creates a clean `torch.utils.data.Subset` filtering out duplicates and leakage in RAM without duplicating datasets on disk.
4. **Contrastive CAM:** Answers *"Why class A instead of class B?"* by backpropagating the logit difference $z_A - z_B$.
5. **Confidence Calibration:** Quantifies ECE/MCE and provides 1-line Temperature Scaling (L-BFGS optimization on validation logits) with proven mathematical invariance for top-1 predictions.
6. **ModelDoctor:** A one-shot diagnostic orchestrator that generates self-contained, dark-mode interactive HTML reports with clickable failure clusters, search bars, and base64-embedded overlays.

We ran a full 10-epoch validation test on Fashion-MNIST with ResNet-18 on GPU: data audit, training, calibration remediation, error clustering, and report generation completed in 21 seconds.

Code and benchmarks are open-source on GitHub. Feedback and contributions are welcome!

Link: https://github.com/csharikrishna/Lensight-
```

---

## 4. Reddit `r/Python` Launch Post

**Title:** `Show r/Python: Lensight – I was tired of installing 5 packages to debug one PyTorch model, so I built a unified, zero-bloat diagnostic toolkit`

```markdown
Hey everyone!

I wanted to share **Lensight**, an open-source Python package I built to streamline quality assurance for computer vision models.

### The Problem
Debugging vision models usually requires:
- `pytorch-grad-cam` for heatmaps
- `cleanlab` for data errors
- `netcal` for calibration
- Custom scripts for duplicate / train-test leakage checks
- Excel / Matplotlib for error reports

Each of these has conflicting dependency trees and different tensor formats.

### The Solution: Lensight
Lensight brings all of this into a single import with a strict dependency diet: **pure PyTorch, NumPy, Pillow, and Scikit-Learn**. No OpenCV, Pandas, or SciPy required.

Quick example:
```python
import lensight
from lensight import ModelDoctor, GradCAM, TemperatureScaler
from lensight.eda import DatasetAuditor, DatasetSanitizer

# 1. Pre-training: Catch train/test leakage & near duplicates
auditor = DatasetAuditor(class_names=["cat", "dog", "fox"])
audit_report = auditor.audit(train_loader)
clean_train_subset = DatasetSanitizer.clean_subset(train_dataset, audit_report)

# 2. Post-training: 1-line full health check & interactive HTML report
doctor = ModelDoctor(model, class_names=["cat", "dog", "fox"])
report = doctor.diagnose(val_loader)
report.save_html("model_health.html")

# 3. Fix overconfidence
scaler = TemperatureScaler(model)
scaler.fit(val_loader)
calibrated_model = scaler.calibrated_model
```

### Performance details:
- Perceptual hashing uses vectorized NumPy bitwise XOR popcount tables ($50\times$ speedup over Python loop approaches).
- Built-in interactive REPL cheat sheet: `lensight.help()` or CLI `lensight help`.
- Wheel package is under 80 KB.
- All 45 unit tests execute in ~8 seconds.

GitHub: https://github.com/csharikrishna/Lensight-  
PyPI: `pip install lensight`

Would love to hear your thoughts, feedback, and feature requests!
```

---

## 5. Twitter / X Launch Thread

```text
🧵 1/6
Excited to announce Lensight: a unified, production-grade PyTorch diagnostic & dataset health toolkit. 🔍⚡

No more installing 5 fragmented packages just to debug one computer vision model.

GitHub: https://github.com/csharikrishna/Lensight-
#PyTorch #ComputerVision #DeepLearning

---

2/6
Computer vision debugging is broken:
• You use one lib for Grad-CAM
• Another lib for label errors
• Another lib for calibration curves
• Hand-rolled scripts for train/test leakage

Lensight unifies the entire vision QA lifecycle under one cohesive API.

---

3/6
Pre-Training Data Health 📊:
• Catches near-duplicates & cross-label contradictions
• Flags train/test split leakage
• 50x faster pure NumPy vectorized bitwise hashing
• "Smart Survivor Selection" preserves highest-sharpness images instead of deleting alphabetically

---

4/6
Model Explainability & Calibration 🧠:
• Native Grad-CAM, Grad-CAM++, HiResCAM & Contrastive CAM ("Why Coat instead of Pullover?")
• Vectorized GPU batch attribution
• Temperature Scaling fixes overconfidence while keeping top-1 accuracy 100% invariant

---

5/6
One-Shot ModelDoctor 🩺:
One line to diagnose everything:
`doctor = ModelDoctor(model, class_names=...)`
`report = doctor.diagnose(val_loader)`

Generates self-contained, interactive dark-mode HTML dashboards with failure clusters & CI/CD JSON metrics.

---

6/6
Zero dependency bloat:
❌ No OpenCV
❌ No Pandas
❌ No SciPy
✅ Pure PyTorch, NumPy, Pillow, Scikit-Learn

Wheel size < 80 KB. 45 tests pass in 8s.
Check out the repo, star ⭐, and let us know what you think:
👉 https://github.com/csharikrishna/Lensight-
```
