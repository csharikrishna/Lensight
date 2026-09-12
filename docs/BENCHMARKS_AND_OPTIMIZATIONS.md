# Benchmarks & Engineering Optimizations

This document provides rigorous, reproducible benchmark comparisons, algorithmic complexity analyses, and engineering optimizations implemented in **Lensight** versus legacy alternatives.

All numbers in this document are directly reproducible using the standalone scripts in the [`benchmarks/`](../benchmarks/) directory.

---

## 1. Executive Summary: What Makes Lensight Different?

Before Lensight, computer vision teams had to patch together 5 to 6 disparate tools:
* `imgcheck` / `imgdedup` for duplicate scanning (slow Python loops, arbitrary deletion).
* `pytorch-grad-cam` / `captum` for visual attribution (single-item loops, high memory overhead).
* `netcal` for calibration (heavy external dependencies).
* `cleanlab` for data curation.
* Custom ad-hoc scripts for error slicing.

Lensight re-engineers this fragmented stack into a cohesive, zero-bloat package with **$12\times$ to $16\times$ faster algorithmic execution**, **$5\times$ to $7.5\times$ batched GPU attribution throughput**, and **zero-disk in-memory workflows**.

---

## 2. Benchmark 1: Pairwise Duplicate & Leakage Auditing

**Runnable Script:** [`benchmarks/bench_dedup_speedup.py`](../benchmarks/bench_dedup_speedup.py)  
**Hardware & Environment:** Intel64 / x86_64, Windows 11, Python 3.12.10, NumPy 2.4.2

### Algorithmic Comparison
* **Legacy Approach (`imgcheck`, `imagehash`)**:
  Iterates through pairs using nested Python loops with individual comparison calls:
  ```python
  # O(N^2) in pure Python interpreter
  for i in range(N):
      for j in range(i + 1, N):
          if hamming_distance(hashes[i], hashes[j]) <= threshold:
              clusters.add(...)
  ```
* **Lensight Vectorized Engine**:
  Bit-unpacks 64-bit perceptual hashes into 8-byte uint8 NumPy arrays and applies a 256-element bitwise XOR popcount lookup table (`_POPCOUNT_8`):
  ```python
  # Vectorized in C / SIMD registers
  diff = A[:, None, :] ^ B[None, :, :]
  dist_mat = _POPCOUNT_8[diff].sum(axis=-1)
  ```

### Measured Performance Benchmark

| Dataset Size | Legacy imgcheck Loop | Lensight Vectorized Engine | **Measured Speedup** |
| :--- | :--- | :--- | :--- |
| **500 images** | 0.167 s | **0.010 s** | **$16.6\times$ faster** |
| **1,500 images** | 1.418 s | **0.087 s** | **$16.3\times$ faster** |
| **2,500 images** | 3.788 s | **0.249 s** | **$15.2\times$ faster** |

> **Reproducibility Note**: Run `python benchmarks/bench_dedup_speedup.py` to reproduce these measurements. Speedup factors consistently measure between **$12\times$ and $17\times$** across varying hardware and NumPy BLAS configurations.

---

## 3. Benchmark 2: Quality-Aware Survivor Selection vs. Deletion Loss

### The Failure of Legacy Selection
Legacy tools (including `imgcheck-0.1.0`) pick the surviving image in a duplicate cluster using the **alphabetically first file path** or arbitrary discovery order:

```python
# Legacy survivor selection
survivor = cluster_members[0] # Alphabetical or os.listdir order
```

**The Danger**: If a dataset contains a high-resolution 4K original (`IMG_9821.JPG`) and a low-resolution thumbnail (`thumb_01.jpg`), legacy tools frequently discard the high-resolution master and retain the compressed thumbnail, degrading the quality of the dataset.

### Lensight Smart Survivor Selection
Lensight evaluates an objective image fidelity metric:

$$Q(x) = \text{Var}\left(\nabla^2 I_{\text{gray}}\right) \times \left(W \times H\right)$$

Using a pure NumPy 2D Laplacian convolution kernel:
$$\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

| Test Scenario | Image A (Original) | Image B (Compressed Copy) | Legacy Tool Keeps | Lensight Keeps |
| :--- | :--- | :--- | :--- | :--- |
| **Resolution Mismatch** | $1024 \times 1024$ (Sharp) | $128 \times 128$ (Blurry) | `a_thumb.jpg` (Blurry) | **Image A (Sharp Original)** |
| **Focus Degradation** | Sharp in-focus sample | Defocused motion blur copy | Arbitrary | **Sharp In-Focus Sample** |
| **Compression Artifacts** | Clean PNG (lossless) | JPEG Quality 30 | Arbitrary | **Clean PNG** |

---

## 4. Benchmark 3: Memory Footprint & Zero-Disk Storage Overhead

**Runnable Script:** [`benchmarks/bench_memory_footprint.py`](../benchmarks/bench_memory_footprint.py)  
**Runtime:** Python 3.12.10, PyTorch 2.5.1

Legacy data cleaning tools physically clone surviving files into a brand-new directory on disk. Lensight uses `DatasetSanitizer.clean_subset()` to instantiate an in-memory `torch.utils.data.Subset` that filters out corrupt, leaked, and duplicate indices with **zero disk writes**.

### Storage Overhead Comparison Across Profiles

#### Profile A: Standard Web / CV Thumbnails (~50 KB / image, e.g., 256×256 ImageNet)
| Dataset Size | Raw Dataset Size | Legacy Sanitization (`imgcheck`) | Lensight In-Memory Sanitization |
| :--- | :--- | :--- | :--- |
| **5,000 images** | 244.1 MB | 219.7 MB duplicated to disk | **0.0 MB written** (RAM: 36.3 KB) |
| **25,000 images** | 1.19 GB | 1.07 GB duplicated to disk | **0.0 MB written** (RAM: 190.1 KB) |
| **100,000 images** | 4.77 GB | 4.29 GB duplicated to disk | **0.0 MB written** (RAM: 782.2 KB) |

#### Profile B: High-Resolution Production Photos (~850 KB / image, e.g., 1080p DSLR / Inspection)
| Dataset Size | Raw Dataset Size | Legacy Sanitization (`imgcheck`) | Lensight In-Memory Sanitization |
| :--- | :--- | :--- | :--- |
| **5,000 images** | 4.05 GB | 3.65 GB duplicated to disk | **0.0 MB written** (RAM: 36.3 KB) |
| **25,000 images** | 20.27 GB | 18.24 GB duplicated to disk | **0.0 MB written** (RAM: 190.1 KB) |
| **100,000 images** | 81.06 GB (~81 GB) | 72.96 GB duplicated to disk | **0.0 MB written** (RAM: 782.2 KB) |

* **Legacy**: Forces users to duplicate the entire dataset on disk before training, generating between 4.3 GB and 73+ GB of redundant I/O writes.
* **Lensight**: Returns an in-memory `torch.utils.data.Subset`. **Zero disk writes, zero storage duplication, and less than 1 MB of index RAM.**

---

## 5. Benchmark 4: Vectorized Batch CAM Throughput

**Runnable Script:** [`benchmarks/bench_cam_throughput.py`](../benchmarks/bench_cam_throughput.py)  
**Hardware & Environment:** NVIDIA GeForce GTX 1650 (CUDA 12.1), PyTorch 2.5.1, Batch Size = 16, Resolution = 64×64

Most CAM packages evaluate explainability image-by-image using sequential Python loops:
```python
for img in batch: # B separate forward and backward passes
    cam.explain(img)
```

Lensight implements batch-vectorized backward attribution in `BaseCAM.explain_batch`:
* A single forward pass produces logits for all $B$ images.
* Target class scores are summed: $S = \sum_{i=1}^B Y_i^{c_i}$.
* A single `S.backward()` computes exact gradients for all $B$ feature maps simultaneously.

### Measured Attribution Throughput on NVIDIA GPU

| Method | Single-Image Loop | Lensight `explain_batch` | **Measured Throughput Gain** |
| :--- | :--- | :--- | :--- |
| **Grad-CAM** | 233.1 imgs/sec | **1,693.3 imgs/sec** | **$7.3\times$ faster** |
| **HiResCAM** | 222.5 imgs/sec | **1,662.3 imgs/sec** | **$7.5\times$ faster** |
| **Grad-CAM++** | 203.6 imgs/sec | **1,125.3 imgs/sec** | **$5.5\times$ faster** |

---

## 6. Dependency Footprint Audit

| Metric | Typical Fragmented Stack | Lensight |
| :--- | :--- | :--- |
| **Direct & Transitive Packages** | 28+ dependencies (OpenCV, Pandas, SciPy, PyWavelets, fpdf2, etc.) | **4 core packages** (`torch`, `numpy`, `pillow`, `scikit-learn`) |
| **Wheel Archive Size** | > 150 MB combined | **73.6 KB** |
| **Automated Test Suite Execution** | ~60+ seconds | **~12–16 seconds** (45 unit tests) |
| **Installation Time (`pip install`)** | ~45–90 seconds | **< 3 seconds** |
