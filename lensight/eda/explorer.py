"""
DatasetExplorer: Lightweight, plug-and-play Computer Vision Exploratory Data Analysis (EDA).

Focuses on the critical data anomalies that cause real-world CV models to fail:
1. Class Imbalance & Representation
2. Exposure & Contrast Defects (underexposed, overexposed, washed out)
3. Blur & Sharpness Estimation (Laplacian gradient energy)
4. Aspect Ratio & Resolution Consistency
5. Exact and Near-Duplicate Detection (Perceptual difference hashing)
"""

from __future__ import annotations
import html
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.utils.data as data

from ..utils.image_utils import _resize_2d


def _compute_laplacian_variance(gray_img: np.ndarray) -> float:
    """Compute variance of the 2D Laplacian as an index of sharpness/focus."""
    h, w = gray_img.shape
    if h < 3 or w < 3:
        return 0.0
    # 3x3 discrete Laplacian filter: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    lap = (
        gray_img[1:-1, 2:]
        + gray_img[1:-1, :-2]
        + gray_img[2:, 1:-1]
        + gray_img[:-2, 1:-1]
        - 4.0 * gray_img[1:-1, 1:-1]
    )
    return float(np.var(lap))


def _compute_dhash(gray_img: np.ndarray) -> int:
    """Compute 64-bit difference hash (dhash) for fast duplicate detection."""
    thumb = _resize_2d(gray_img, 8, 9)
    diff = thumb[:, 1:] > thumb[:, :-1]
    # Pack 64 booleans into a single 64-bit int
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


@dataclass
class EDAReport:
    total_images: int
    class_names: list[str] | None
    class_counts: dict[int, int]
    imbalance_ratio: float
    is_imbalanced: bool

    # Exposure & Contrast
    mean_brightness: float
    mean_contrast: float
    underexposed_indices: list[int]
    overexposed_indices: list[int]
    low_contrast_indices: list[int]

    # Sharpness & Blur
    mean_sharpness: float
    blurriest_indices: list[int]

    # Dimensions & Channels
    shapes: list[tuple[int, ...]]
    aspect_ratios: list[float]
    is_variable_size: bool

    # Duplicates
    duplicate_pairs: list[tuple[int, int]]

    # Stored representative outlier images for display (index, tag, HWC uint8 array)
    outlier_images: list[tuple[int, str, np.ndarray]] = field(default_factory=list)

    def label_name(self, idx: int) -> str:
        if self.class_names and 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f"Class {idx}"

    def summary_text(self) -> str:
        lines = [
            "==================================================",
            "  LENSIGHT: DATASET HEALTH & EXPLORATORY AUDIT",
            "==================================================",
            f"Total Samples Analyzed: {self.total_images}",
            f"Class Imbalance Ratio:  {self.imbalance_ratio:.2f}x "
            + ("[ALERT: Imbalanced!]" if self.is_imbalanced else "[OK: Balanced]"),
        ]

        if self.class_counts:
            lines.append("Class Distribution:")
            for cls_idx, count in sorted(self.class_counts.items(), key=lambda x: -x[1]):
                pct = (count / self.total_images) * 100 if self.total_images else 0
                lines.append(f"  - {self.label_name(cls_idx)}: {count} ({pct:.1f}%)")

        lines.extend([
            f"\nVisual Exposure & Contrast:",
            f"  - Mean Brightness:     {self.mean_brightness:.1%}",
            f"  - Mean Contrast (std): {self.mean_contrast:.3f}",
            f"  - Underexposed (dark): {len(self.underexposed_indices)} images",
            f"  - Overexposed (blown): {len(self.overexposed_indices)} images",
            f"  - Low Contrast (flat): {len(self.low_contrast_indices)} images",
            f"\nSharpness & Focus:",
            f"  - Mean Sharpness:      {self.mean_sharpness:.4f}",
            f"  - Suspected Blurry:    {len(self.blurriest_indices)} images",
            f"\nIntegrity & Leakage:",
            f"  - Duplicate Pairs:     {len(self.duplicate_pairs)} pairs",
            f"  - Variable Sizes:      {'Yes' if self.is_variable_size else 'No (Uniform)'}",
        ])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "class_distribution": {
                self.label_name(k): v for k, v in self.class_counts.items()
            },
            "imbalance_ratio": self.imbalance_ratio,
            "is_imbalanced": self.is_imbalanced,
            "exposure": {
                "mean_brightness": self.mean_brightness,
                "mean_contrast": self.mean_contrast,
                "underexposed_count": len(self.underexposed_indices),
                "overexposed_count": len(self.overexposed_indices),
                "low_contrast_count": len(self.low_contrast_indices),
            },
            "sharpness": {
                "mean_sharpness": self.mean_sharpness,
                "blurry_count": len(self.blurriest_indices),
            },
            "integrity": {
                "duplicate_pairs_count": len(self.duplicate_pairs),
                "is_variable_size": self.is_variable_size,
            },
        }

    def save_html(self, path: str, title: str = "Dataset Exploratory Analysis (EDA)") -> None:
        from .eda_report import build_eda_html_report
        html_str = build_eda_html_report(self, title=title)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_str)

    def _repr_html_(self) -> str:
        from .eda_report import build_eda_html_report
        html_str = build_eda_html_report(self)
        escaped = html.escape(html_str)
        return (
            f'<iframe style="width: 100%; height: 750px; border: 1px solid #cbd5e1; '
            f'border-radius: 8px;" srcdoc="{escaped}"></iframe>'
        )


class DatasetExplorer:
    """
    Plug-and-play Computer Vision Dataset Profiler.

    Usage
    -----
        explorer = DatasetExplorer(val_dataloader, class_names=classes)
        report = explorer.analyze(max_samples=1000)
        print(report.summary_text())
        report.save_html("dataset_eda.html")
    """

    def __init__(
        self,
        data_source: Any,
        class_names: Sequence[str] | None = None,
        store_outliers: int = 8,
    ):
        self.data_source = data_source
        self.class_names = list(class_names) if class_names else None
        self.store_outliers = store_outliers

    def analyze(self, max_samples: int | None = 1000) -> EDAReport:
        """
        Execute streaming dataset analysis across images and labels.

        Parameters
        ----------
        max_samples : int, optional
            Cap on the number of samples to profile for rapid analysis. Defaults to 1000.
        """
        loader = self._prepare_loader()

        class_counts: dict[int, int] = {}
        brightnesses: list[float] = []
        contrasts: list[float] = []
        sharpnesses: list[float] = []
        aspect_ratios: list[float] = []
        shapes_set: set[tuple[int, ...]] = set()

        hashes: dict[int, list[int]] = {}  # hash_val -> list of sample indices
        underexposed: list[int] = []
        overexposed: list[int] = []
        low_contrast: list[int] = []

        sample_idx = 0
        outlier_candidates: list[tuple[int, str, np.ndarray, float]] = []

        for batch in loader:
            imgs = batch[0]
            lbls = batch[1] if len(batch) > 1 else None

            # Convert batch to list of samples
            batch_size = imgs.shape[0] if isinstance(imgs, (torch.Tensor, np.ndarray)) else len(imgs)

            for i in range(batch_size):
                if max_samples and sample_idx >= max_samples:
                    break

                img = imgs[i]
                lbl = int(lbls[i].item()) if lbls is not None and hasattr(lbls[i], "item") else (int(lbls[i]) if lbls is not None else -1)

                if lbl >= 0:
                    class_counts[lbl] = class_counts.get(lbl, 0) + 1

                # Normalize image to HWC float32 in [0, 1]
                arr = self._to_hwc_float(img)
                h, w = arr.shape[:2]
                shapes_set.add(arr.shape)
                aspect_ratios.append(h / max(1, w))

                # Grayscale
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
                elif arr.ndim == 3 and arr.shape[-1] == 1:
                    gray = arr[..., 0]
                else:
                    gray = arr

                b_val = float(np.mean(gray))
                c_val = float(np.std(gray))
                s_val = _compute_laplacian_variance(gray)
                d_hash = _compute_dhash(gray)

                brightnesses.append(b_val)
                contrasts.append(c_val)
                sharpnesses.append(s_val)

                # Duplicate indexing
                hashes.setdefault(d_hash, []).append(sample_idx)

                # Flag defects
                uint8_img = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                if uint8_img.ndim == 2:
                    uint8_img = np.repeat(uint8_img[..., None], 3, axis=-1)
                elif uint8_img.shape[-1] == 1:
                    uint8_img = np.repeat(uint8_img, 3, axis=-1)

                if b_val < 0.12:
                    underexposed.append(sample_idx)
                    outlier_candidates.append((sample_idx, "Underexposed (Dark)", uint8_img, b_val))
                elif b_val > 0.88:
                    overexposed.append(sample_idx)
                    outlier_candidates.append((sample_idx, "Overexposed (Blown)", uint8_img, -b_val))
                elif c_val < 0.04:
                    low_contrast.append(sample_idx)
                    outlier_candidates.append((sample_idx, "Low Contrast (Washed)", uint8_img, c_val))

                sample_idx += 1

            if max_samples and sample_idx >= max_samples:
                break

        total = sample_idx

        # Class Imbalance
        if class_counts:
            min_c = min(class_counts.values())
            max_c = max(class_counts.values())
            imbalance_ratio = (max_c / min_c) if min_c > 0 else float(max_c)
            is_imbalanced = imbalance_ratio >= 3.0
        else:
            imbalance_ratio = 1.0
            is_imbalanced = False

        # Blurriest images (lowest 5% sharpness)
        if sharpnesses:
            sorted_by_sharpness = np.argsort(sharpnesses)
            n_blurry = max(1, int(len(sharpnesses) * 0.05)) if min(sharpnesses) < 1e-4 else 0
            blurriest = [int(idx) for idx in sorted_by_sharpness[:n_blurry]]
        else:
            blurriest = []

        # Duplicate pairs
        duplicate_pairs: list[tuple[int, int]] = []
        for h_val, idxs in hashes.items():
            if len(idxs) > 1:
                for j in range(len(idxs) - 1):
                    duplicate_pairs.append((idxs[j], idxs[j + 1]))

        # Select stored outlier gallery images
        stored_outliers: list[tuple[int, str, np.ndarray]] = []
        for idx, tag, o_img, _ in outlier_candidates[: self.store_outliers]:
            stored_outliers.append((idx, tag, o_img))

        return EDAReport(
            total_images=total,
            class_names=self.class_names,
            class_counts=class_counts,
            imbalance_ratio=imbalance_ratio,
            is_imbalanced=is_imbalanced,
            mean_brightness=float(np.mean(brightnesses)) if brightnesses else 0.0,
            mean_contrast=float(np.mean(contrasts)) if contrasts else 0.0,
            underexposed_indices=underexposed,
            overexposed_indices=overexposed,
            low_contrast_indices=low_contrast,
            mean_sharpness=float(np.mean(sharpnesses)) if sharpnesses else 0.0,
            blurriest_indices=blurriest,
            shapes=list(shapes_set),
            aspect_ratios=aspect_ratios,
            is_variable_size=(len(shapes_set) > 1),
            duplicate_pairs=duplicate_pairs,
            outlier_images=stored_outliers,
        )

    def _prepare_loader(self):
        if isinstance(self.data_source, data.DataLoader):
            return self.data_source
        if isinstance(self.data_source, data.Dataset):
            return data.DataLoader(self.data_source, batch_size=32, shuffle=False)
        return self.data_source

    @staticmethod
    def _to_hwc_float(img: Any) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().float().numpy()
        else:
            arr = np.asarray(img, dtype=np.float32)

        # Handle CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))

        # Normalize to [0, 1]
        if arr.max() > 1.5:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)
