"""
Benchmark 2: Vectorized Batch CAM Throughput vs Single-Image Loop.
Reproduces Section 5 of BENCHMARKS_AND_OPTIMIZATIONS.md.

Compares:
1. Single-image loop: `for img in batch: cam.explain(img)`
2. Lensight vectorized batching: `cam.explain_batch(batch)`

Supports both CPU and GPU (NVIDIA CUDA / Apple MPS).
"""
import platform
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from lensight.cam.gradcam import GradCAM
from lensight.cam.hirescam import HiResCAM
from lensight.cam.gradcam_plus_plus import GradCAMPlusPlus

class BenchmarkConvNet(nn.Module):
    """Standard multi-layer CNN for benchmark reproducibility without downloading remote weights."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def print_system_specs(device):
    print("=" * 70)
    print("SYSTEM AND RUNTIME SPECIFICATIONS:")
    print(f"  OS:             {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  PyTorch Version:{torch.__version__}")
    print(f"  Active Device:  {device}")
    if device.type == "cuda":
        print(f"  GPU Name:       {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version:   {torch.version.cuda}")
    print("=" * 70)

def benchmark_method(name, explainer, batch, single_loop_fn, batch_fn, warmup=2, runs=5):
    # Warmup
    for _ in range(warmup):
        single_loop_fn()
        batch_fn()

    # Time single loop
    t0 = time.perf_counter()
    for _ in range(runs):
        single_loop_fn()
    t_single = (time.perf_counter() - t0) / runs

    # Time batch
    t0 = time.perf_counter()
    for _ in range(runs):
        batch_fn()
    t_batch = (time.perf_counter() - t0) / runs

    batch_size = batch.shape[0]
    fps_single = batch_size / t_single
    fps_batch = batch_size / t_batch
    speedup = fps_batch / fps_single

    print(f"{name:<20} | {fps_single:>12.1f} imgs/s | {fps_batch:>14.1f} imgs/s | {speedup:>10.1f}x")

def run_benchmark(batch_size=16, img_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_specs(device)
    print(f"\nConfiguration: Batch Size = {batch_size}, Image Dimension = (3, {img_size}, {img_size})\n")
    print(f"{'Method':<20} | {'Single Loop':>19} | {'explain_batch':>21} | {'Throughput':>11}")
    print("-" * 75)

    model = BenchmarkConvNet().to(device)
    model.eval()

    # Target convolution layer
    target_layer = model.features[8]  # Conv2d(64, 128, ...)

    batch = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # 1. Grad-CAM
    gcam = GradCAM(model, target_layer=target_layer)
    benchmark_method(
        "Grad-CAM",
        gcam,
        batch,
        single_loop_fn=lambda: [gcam.explain(batch[i:i+1], target_class=0) for i in range(batch_size)],
        batch_fn=lambda: gcam.explain_batch(batch, target_classes=[0] * batch_size),
    )

    # 2. HiResCAM
    hcam = HiResCAM(model, target_layer=target_layer)
    benchmark_method(
        "HiResCAM",
        hcam,
        batch,
        single_loop_fn=lambda: [hcam.explain(batch[i:i+1], target_class=0) for i in range(batch_size)],
        batch_fn=lambda: hcam.explain_batch(batch, target_classes=[0] * batch_size),
    )

    # 3. Grad-CAM++
    gcam_pp = GradCAMPlusPlus(model, target_layer=target_layer)
    benchmark_method(
        "Grad-CAM++",
        gcam_pp,
        batch,
        single_loop_fn=lambda: [gcam_pp.explain(batch[i:i+1], target_class=0) for i in range(batch_size)],
        batch_fn=lambda: gcam_pp.explain_batch(batch, target_classes=[0] * batch_size),
    )

    print("-" * 75)
    print(" CAM Batch Throughput Benchmark completed successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lensight CAM Throughput Benchmark")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size to benchmark")
    parser.add_argument("--img-size", type=int, default=64, help="Image resolution")
    args = parser.parse_args()
    run_benchmark(batch_size=args.batch_size, img_size=args.img_size)
