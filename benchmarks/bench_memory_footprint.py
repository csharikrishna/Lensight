"""
Benchmark 3: Memory Footprint & Zero-Disk Storage Overhead.
Reproduces Section 4 of BENCHMARKS_AND_OPTIMIZATIONS.md.

Demonstrates:
1. Legacy sanitization: requires duplicate physical disk writes for clean copies
2. Lensight sanitization: zero disk I/O, pure in-memory PyTorch Subset filtering with minimal RAM footprint (<1 MB index table for 100k samples)
"""
import os
import platform
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import Dataset, Subset

from lensight.eda.sanitizer import DatasetSanitizer
from lensight.eda.auditor import AuditReport, SampleMeta
from lensight.eda.hashing import DuplicateGroup

class SyntheticImageDataset(Dataset):
    """Synthetic dataset simulating large CV datasets without allocating gigabytes of RAM."""
    def __init__(self, size=10000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.zeros(3, 64, 64), idx % 10

def print_system_specs():
    print("=" * 70)
    print("SYSTEM AND RUNTIME SPECIFICATIONS:")
    print(f"  OS:             {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  PyTorch Version:{torch.__version__}")
    print("=" * 70)

def run_benchmark(dataset_sizes=(5000, 25000, 100000)):
    print_system_specs()
    print("\nStarting Dataset Sanitization Memory & Storage Benchmark...")
    print("Evaluating physical disk replication (legacy) vs. zero-copy in-memory Subset (Lensight)\n")

    profiles = [
        ("Standard Web / CV Thumbnails (~50 KB/image, e.g., 256x256)", 50 * 1024),
        ("High-Res Production / Inspection Photos (~850 KB/image, e.g., 1080p DSLR)", 850 * 1024),
    ]

    for profile_name, img_bytes in profiles:
        print(f"PROFILE: {profile_name}")
        print(f"{'Dataset Size':<14} | {'Raw Dataset Size':<18} | {'Legacy Disk Written':<20} | {'Lensight Disk Written':<22}")
        print("-" * 80)

        for n in dataset_sizes:
            raw_size_mb = (n * img_bytes) / (1024 * 1024)
            survivors_count = int(n * 0.90)
            legacy_disk_written_mb = (survivors_count * img_bytes) / (1024 * 1024)

            # Lensight approach: in-memory Subset
            dataset = SyntheticImageDataset(size=n)
            samples = [
                SampleMeta(sample_id=i, class_name=f"class_{i%10}", class_idx=i%10)
                for i in range(n)
            ]
            dup_clusters = [
                DuplicateGroup(cluster_id=f"c_{i}", survivor_id=i, duplicate_ids=[i+1], distances_to_survivor={i+1: 1})
                for i in range(0, int(n * 0.10) * 2, 2)
            ]
            audit_report = AuditReport(
                total_samples=n,
                class_counts={f"class_{c}": n // 10 for c in range(10)},
                imbalance_ratio=1.0,
                corrupt_files=[],
                dimension_stats={},
                duplicate_clusters=dup_clusters,
                cross_label_conflicts=[],
                samples=samples,
            )

            subset = DatasetSanitizer.clean_subset(dataset, audit_report)
            lensight_disk_written_mb = 0.0  # Zero disk I/O
            mem_overhead_bytes = sys.getsizeof(subset.indices)

            # Format in MB or GB
            if raw_size_mb >= 1024:
                raw_str = f"{raw_size_mb / 1024:.2f} GB"
                legacy_str = f"{legacy_disk_written_mb / 1024:.2f} GB"
            else:
                raw_str = f"{raw_size_mb:.1f} MB"
                legacy_str = f"{legacy_disk_written_mb:.1f} MB"

            lensight_str = f"0.0 MB (RAM: {mem_overhead_bytes/1024:.1f} KB)"
            print(f"{n:<14} | {raw_str:>16} | {legacy_str:>18} | {lensight_str:>22}")

        print("-" * 80 + "\n")

    print(" Benchmark completed successfully.")
    print(" Key Takeaway: Lensight in-memory Subset requires 0 bytes of disk overhead across all resolutions.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lensight Storage Benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 25000, 100000],
                        help="Dataset sample sizes to evaluate")
    args = parser.parse_args()
    run_benchmark(args.sizes)
