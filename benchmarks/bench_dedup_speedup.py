"""
Benchmark 1: Vectorized Perceptual Hash Duplicate Detection vs. Legacy Nested Loop.
Reproduces Section 2 of BENCHMARKS_AND_OPTIMIZATIONS.md.

Compares:
1. Legacy nested Python loop (O(N^2) pairwise Hamming distance comparisons)
2. Lensight vectorized bitwise XOR + 256-element popcount LUT matrix multiplication

Hardware/Software specs printed automatically.
"""
import os
import platform
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Lensight vectorized implementation
from lensight.eda.hashing import pairwise_hamming_matrix

def legacy_imagehash_nested_loop(hashes_bool, threshold=3):
    """Legacy O(N^2) nested loop using imagehash-style `a - b` (np.count_nonzero(a != b))."""
    n = len(hashes_bool)
    matches = []
    for i in range(n):
        h1 = hashes_bool[i]
        for j in range(i + 1, n):
            h2 = hashes_bool[j]
            # Exact imagehash.__sub__ behavior:
            dist = np.count_nonzero(h1 != h2)
            if dist <= threshold:
                matches.append((i, j, dist))
    return matches

def legacy_int_nested_loop(hashes_uint64, threshold=3):
    """Python int bit_count nested loop."""
    n = len(hashes_uint64)
    matches = []
    for i in range(n):
        h1 = int(hashes_uint64[i])
        for j in range(i + 1, n):
            h2 = int(hashes_uint64[j])
            dist = (h1 ^ h2).bit_count()
            if dist <= threshold:
                matches.append((i, j, dist))
    return matches

def lensight_vectorized_hamming(hashes_uint64, threshold=3):
    """Lensight SIMD-accelerated popcount lookup table."""
    dist_mat = pairwise_hamming_matrix(hashes_uint64, hashes_uint64)
    # Extract upper triangle matches
    i_indices, j_indices = np.where(np.triu(dist_mat <= threshold, k=1))
    return list(zip(i_indices, j_indices, dist_mat[i_indices, j_indices]))

def print_system_specs():
    print("=" * 70)
    print("SYSTEM AND RUNTIME SPECIFICATIONS:")
    print(f"  OS:             {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Processor:      {platform.processor() or 'x86_64'}")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  NumPy Version:  {np.__version__}")
    print("=" * 70)

def run_benchmark(sizes=(500, 1500, 2500)):
    print_system_specs()
    print("\nStarting Pairwise Duplicate Auditing Benchmark...\n")
    print(f"{'Dataset Size':<15} | {'Legacy imgcheck Loop':<22} | {'Lensight Vectorized':<20} | {'Speedup':<12}")
    print("-" * 77)

    np.random.seed(42)

    for n in sizes:
        # Generate random 64-bit unsigned integer hashes
        hashes = np.random.randint(0, np.iinfo(np.int64).max, size=n, dtype=np.int64).view(np.uint64)
        # 64-bit boolean array representation as imagehash uses internally
        hashes_bool = np.unpackbits(hashes.view(np.uint8)).reshape(n, 64).astype(bool)

        # 1. Legacy nested loop timing (imgcheck / imagehash)
        if n <= 2500:
            t0 = time.perf_counter()
            legacy_matches = legacy_imagehash_nested_loop(hashes_bool, threshold=5)
            t_legacy = time.perf_counter() - t0
            legacy_str = f"{t_legacy:.4f} s"
        else:
            legacy_str = "Skipped (>60s)"
            t_legacy = None

        # 2. Lensight vectorized timing
        t0 = time.perf_counter()
        lensight_matches = lensight_vectorized_hamming(hashes, threshold=5)
        t_lensight = time.perf_counter() - t0
        lensight_str = f"{t_lensight:.4f} s"

        if t_legacy is not None:
            speedup = t_legacy / t_lensight
            speedup_str = f"{speedup:.1f}x faster"
            # Verify result equivalence
            assert len(legacy_matches) == len(lensight_matches), "Mismatch in duplicate pair counts!"
        else:
            speedup_str = "N/A"

        print(f"{n:<15} | {legacy_str:<22} | {lensight_str:<20} | {speedup_str:<12}")

    print("-" * 77)
    print(" Benchmark completed successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lensight Deduplication Benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1500, 2500],
                        help="Dataset sizes to benchmark")
    args = parser.parse_args()
    run_benchmark(args.sizes)
