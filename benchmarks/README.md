# Lensight Benchmark Suite

This directory contains standalone, reproducible benchmark scripts corresponding to the findings in [`docs/BENCHMARKS_AND_OPTIMIZATIONS.md`](../docs/BENCHMARKS_AND_OPTIMIZATIONS.md).

Every script runs in seconds on commodity developer hardware (CPU or GPU) without requiring proprietary datasets or network downloads.

---

## 📋 Benchmark Scripts

### 1. Vectorized Deduplication Speedup (`bench_dedup_speedup.py`)
Reproduces the **12x to 16x speedup** achieved by Lensight's bitwise XOR popcount lookup table (`pairwise_hamming_matrix`) compared to legacy nested Python loops (`imgcheck`, `imagehash`).

**To run:**
```bash
python benchmarks/bench_dedup_speedup.py
```

**Options:**
```bash
python benchmarks/bench_dedup_speedup.py --sizes 500 1500 2500
```

---

### 2. Vectorized Batch CAM Throughput (`bench_cam_throughput.py`)
Measures visual attribution throughput (images/sec) comparing single-image sequential loops against Lensight's batched backpropagation (`explain_batch`). Demonstrates **5.5x to 7.5x throughput gains** on CNN architectures.

**To run:**
```bash
python benchmarks/bench_cam_throughput.py
```

**Options:**
```bash
python benchmarks/bench_cam_throughput.py --batch-size 32 --img-size 64
```

---

### 3. Memory & Zero-Disk Storage Overhead (`bench_memory_footprint.py`)
Demonstrates how Lensight's non-destructive in-memory `DatasetSanitizer.sanitize_subset()` avoids gigabytes of duplicate disk writes compared to legacy file-copying tools.

**To run:**
```bash
python benchmarks/bench_memory_footprint.py
```

---

## 💻 Hardware & Environment Verification

When any benchmark script is executed, it automatically prints the active runtime specifications:
* Operating System & Architecture
* Python Version
* NumPy and PyTorch Versions
* Compute Accelerator (CUDA GPU model and driver version, or CPU SIMD)
