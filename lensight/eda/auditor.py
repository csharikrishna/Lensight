"""Comprehensive dataset auditing engine: Leakage detection, Cross-label conflicts, and Transitive duplicates.

Supports both filesystem directory trees (class folders on disk) and PyTorch
DataLoaders/Datasets with zero external heavy dependencies.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from PIL import Image

from .hashing import (
    DuplicateGroup,
    cluster_transitive_duplicates,
    compute_hash,
    hamming_distance,
    pairwise_hamming_matrix,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _image_to_base64_thumb(img_input: Any, max_size: int = 140) -> Optional[str]:
    """Convert an image (path, PIL, numpy, or torch) to a base64 JPEG thumbnail data URI."""
    try:
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                return None
            with Image.open(img_input) as im:
                im = im.convert("RGB")
                im.thumbnail((max_size, max_size))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=80)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        elif isinstance(img_input, Image.Image):
            im = img_input.convert("RGB")
            im.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        elif hasattr(img_input, "cpu"):  # PyTorch tensor
            t = img_input.detach().cpu()
            if t.ndim == 4:
                t = t.squeeze(0)
            arr = t.numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)
            if arr.dtype in (np.float32, np.float64):
                if arr.max() <= 1.05:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr).convert("RGB")
            im.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        elif isinstance(img_input, np.ndarray):
            arr = img_input
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)
            if arr.dtype in (np.float32, np.float64):
                if arr.max() <= 1.05:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr).convert("RGB")
            im.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        return None
    except Exception:
        return None


def _calc_laplacian_sharpness(img_gray: np.ndarray) -> float:
    """Compute variance of 3x3 discrete Laplacian filter as sharpness metric."""
    h, w = img_gray.shape
    if h < 3 or w < 3:
        return 0.0
    # 3x3 Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    lap = (
        img_gray[0 : h - 2, 1 : w - 1]
        + img_gray[2:h, 1 : w - 1]
        + img_gray[1 : h - 1, 0 : w - 2]
        + img_gray[1 : h - 1, 2:w]
        - 4.0 * img_gray[1 : h - 1, 1 : w - 1]
    )
    return float(np.var(lap))


@dataclass
class SampleMeta:
    """Metadata recorded for an audited image sample."""
    sample_id: Any  # file path string or integer index
    class_name: str
    class_idx: int
    split: str = "dataset"
    hash_val: int = 0
    width: int = 0
    height: int = 0
    channels: int = 1
    sharpness: float = 0.0
    mean_intensity: float = 0.0
    file_path: Optional[str] = None
    thumb_b64: Optional[str] = None


@dataclass
class CorruptFileInfo:
    """Details of a file that failed to open or parse."""
    file_path: str
    class_name: str
    error: str


@dataclass
class LeakageMatch:
    """A test/validation sample leaked into the training set."""
    test_id: Any
    test_class: str
    train_id: Any
    train_class: str
    hamming_distance: int
    is_exact: bool
    test_thumb: Optional[str] = None
    train_thumb: Optional[str] = None


@dataclass
class CrossLabelConflict:
    """An image appearing under two conflicting class labels."""
    sample_a_id: Any
    class_a: str
    sample_b_id: Any
    class_b: str
    hamming_distance: int
    is_exact: bool
    thumb_a: Optional[str] = None
    thumb_b: Optional[str] = None


@dataclass
class LeakageReport:
    """Report detailing train/test data leakage."""
    train_count: int
    test_count: int
    leaked_count: int
    per_class_leakage: Dict[str, Dict[str, Any]]
    matches: List[LeakageMatch]
    threshold: int
    algo: str

    @property
    def leakage_rate(self) -> float:
        return (self.leaked_count / self.test_count) if self.test_count > 0 else 0.0

    def summary_text(self) -> str:
        lines = [
            "============================================================",
            "                 TRAIN/TEST LEAKAGE AUDIT                   ",
            "============================================================",
            f"Algorithm: {self.algo} (Hamming threshold <= {self.threshold})",
            f"Train samples: {self.train_count}  |  Test samples: {self.test_count}",
            f"Leaked test samples: {self.leaked_count} ({self.leakage_rate * 100:.2f}% of test set)",
        ]
        if self.leaked_count > 0:
            lines.append("\nPer-class leakage breakdown:")
            for cls, stats in sorted(self.per_class_leakage.items()):
                rate = stats.get("leakage_rate", 0.0) * 100
                leaked = stats.get("leaked", 0)
                tot = stats.get("test_count", 0)
                lines.append(f"  * {cls:15s}: {leaked:4d} / {tot:4d} leaked ({rate:5.1f}%)")
            lines.append(f"\nTop {min(5, len(self.matches))} leaked pairs:")
            for m in self.matches[:5]:
                t_name = os.path.basename(str(m.test_id))
                tr_name = os.path.basename(str(m.train_id))
                exact_tag = "[EXACT]" if m.is_exact else f"[dist={m.hamming_distance}]"
                lines.append(f"  * Test '{t_name}' ({m.test_class}) ~ Train '{tr_name}' {exact_tag}")
        else:
            lines.append("\n[OK] No data leakage detected between train and test sets.")
        lines.append("============================================================")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "train_count": self.train_count,
            "test_count": self.test_count,
            "leaked_count": self.leaked_count,
            "leakage_rate": round(self.leakage_rate, 4),
            "threshold": self.threshold,
            "algo": self.algo,
            "per_class_leakage": self.per_class_leakage,
            "matches": [
                {
                    "test_id": str(m.test_id),
                    "test_class": m.test_class,
                    "train_id": str(m.train_id),
                    "train_class": m.train_class,
                    "hamming_distance": m.hamming_distance,
                    "is_exact": m.is_exact,
                }
                for m in self.matches
            ],
        }

    def get_clean_test_indices(self) -> List[Any]:
        """Return list of test identifiers/indices that are NOT leaked."""
        leaked_ids = {m.test_id for m in self.matches}
        # If test_ids are integers
        return [i for i in range(self.test_count) if i not in leaked_ids]

    def to_json(self, output_path: str) -> None:
        """Export leakage metrics and matches to a JSON file."""
        import json
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_html(self, output_path: str, title: str = "Lensight Train/Test Leakage Audit") -> str:
        """Generate and save interactive HTML dashboard for leakage audit."""
        import html
        from datetime import datetime

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        badge_cls = "badge-danger" if self.leaked_count > 0 else "badge-success"
        leak_pct = self.leakage_rate * 100.0

        rows = "".join(
            f"<tr><td><strong>{html.escape(c)}</strong></td>"
            f"<td>{data['leaked']} / {data['test_count']}</td>"
            f"<td>{data['leakage_rate'] * 100:.1f}%</td></tr>"
            for c, data in sorted(self.per_class_leakage.items(), key=lambda x: -x[1]["leakage_rate"])
        )

        match_cards = "".join(
            f'<div style="background:#1f2937;padding:12px;border-radius:6px;border:1px solid #374151;">'
            f'<strong>Test #{m.test_id} ({m.test_class})</strong> &harr; <strong>Train #{m.train_id} ({m.train_class})</strong>'
            f'<div style="font-size:0.85em;color:#9ca3af;margin-top:4px;">Distance: {m.hamming_distance} {"(EXACT MATCH)" if m.is_exact else ""}</div>'
            f'</div>'
            for m in self.matches[:30]
        )

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #111827; color: #f9fafb; margin: 0; padding: 24px; }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ color: #60a5fa; margin-bottom: 4px; }}
    .subtitle {{ color: #9ca3af; font-size: 0.9em; margin-bottom: 24px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .metric-card {{ background: #1f2937; padding: 16px; border-radius: 8px; border: 1px solid #374151; }}
    .metric-card .val {{ font-size: 1.8em; font-weight: bold; color: {'#f87171' if self.leaked_count > 0 else '#34d399'}; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #1f2937; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #374151; }}
    th {{ background: #2d3748; font-weight: 600; color: #d1d5db; }}
    .matches-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-top: 12px; }}
  </style>
</head>
<body>
<div class="container">
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">Generated on {now_str} &bull; Algorithm: {self.algo} (Threshold: {self.threshold})</div>
  <div class="metric-grid">
    <div class="metric-card"><div>Train Samples</div><div class="val" style="color:#60a5fa">{self.train_count}</div></div>
    <div class="metric-card"><div>Test Samples</div><div class="val" style="color:#60a5fa">{self.test_count}</div></div>
    <div class="metric-card"><div>Leaked Samples</div><div class="val">{self.leaked_count}</div></div>
    <div class="metric-card"><div>Leakage Rate</div><div class="val">{leak_pct:.1f}%</div></div>
  </div>
  <h3>Per-Class Leakage Breakdown</h3>
  <table>
    <thead><tr><th>Class</th><th>Leaked / Total</th><th>Leakage Rate</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h3 style="margin-top:28px;">Detected Leakage Matches (Top {min(30, len(self.matches))})</h3>
  <div class="matches-grid">{match_cards or '<div style="color:#9ca3af">No leakage matches detected.</div>'}</div>
</div>
</body>
</html>"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return html_doc



@dataclass
class CrossLabelReport:
    """Report detailing cross-label contradictory duplicate images."""
    total_samples: int
    num_classes: int
    conflict_count: int
    conflicts: List[CrossLabelConflict]
    conflicted_class_pairs: Dict[Tuple[str, str], int]
    threshold: int
    algo: str

    def summary_text(self) -> str:
        lines = [
            "============================================================",
            "             CROSS-LABEL CONTRADICTION AUDIT                ",
            "============================================================",
            f"Algorithm: {self.algo} (Hamming threshold <= {self.threshold})",
            f"Total samples: {self.total_samples} across {self.num_classes} classes",
            f"Conflicting duplicate pairs: {self.conflict_count}",
        ]
        if self.conflict_count > 0:
            lines.append("\nMost conflicted class pairs:")
            for (ca, cb), count in sorted(self.conflicted_class_pairs.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  * '{ca}' <--> '{cb}': {count} contradictory pairs")
            lines.append(f"\nTop {min(5, len(self.conflicts))} contradictory pairs:")
            for c in self.conflicts[:5]:
                a_name = os.path.basename(str(c.sample_a_id))
                b_name = os.path.basename(str(c.sample_b_id))
                exact = "[EXACT]" if c.is_exact else f"[dist={c.hamming_distance}]"
                lines.append(f"  * '{a_name}' ({c.class_a}) vs '{b_name}' ({c.class_b}) {exact}")
        else:
            lines.append("\n[OK] No cross-label duplicate contradictions found.")
        lines.append("============================================================")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "conflict_count": self.conflict_count,
            "threshold": self.threshold,
            "algo": self.algo,
            "conflicts": [
                {
                    "sample_a_id": str(c.sample_a_id),
                    "class_a": c.class_a,
                    "sample_b_id": str(c.sample_b_id),
                    "class_b": c.class_b,
                    "hamming_distance": c.hamming_distance,
                    "is_exact": c.is_exact,
                }
                for c in self.conflicts
            ],
        }


@dataclass
class AuditReport:
    """Complete dataset audit report: duplicates, leakage, cross-labels, and stats."""
    total_samples: int
    class_counts: Dict[str, int]
    imbalance_ratio: float
    corrupt_files: List[CorruptFileInfo]
    dimension_stats: Dict[str, Any]
    duplicate_clusters: List[DuplicateGroup]
    cross_label_conflicts: List[CrossLabelConflict]
    leakage_matches: List[LeakageMatch] = field(default_factory=list)
    samples: List[SampleMeta] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    threshold: int = 3
    algo: str = "dhash"

    @property
    def total_duplicates_removed_count(self) -> int:
        return sum(len(c.duplicate_ids) for c in self.duplicate_clusters)

    def summary_text(self) -> str:
        lines = [
            "============================================================",
            "                 LENSIGHT DATASET AUDIT                     ",
            "============================================================",
            f"Total Valid Images Profiled: {self.total_samples}",
            f"Classes: {len(self.class_counts)}  |  Imbalance Ratio: {self.imbalance_ratio:.2f}x",
        ]
        if self.corrupt_files:
            lines.append(f"Corrupt / Unreadable Files: {len(self.corrupt_files)}")

        lines.append(f"\nDuplicate Clusters (within-class & global): {len(self.duplicate_clusters)}")
        lines.append(f"  * Total redundant duplicate images: {self.total_duplicates_removed_count}")

        if self.cross_label_conflicts:
            lines.append(f"\n[ALERT] Cross-Label Contradictions: {len(self.cross_label_conflicts)} pairs")
            for c in self.cross_label_conflicts[:3]:
                a_name = os.path.basename(str(c.sample_a_id))
                b_name = os.path.basename(str(c.sample_b_id))
                lines.append(f"    * '{a_name}' ({c.class_a}) <--> '{b_name}' ({c.class_b})")

        if self.leakage_matches:
            lines.append(f"\n[ALERT] Train/Test Leakage: {len(self.leakage_matches)} leaked test images")

        if self.warnings:
            lines.append("\nActionable Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        else:
            lines.append("\n[OK] Dataset is healthy! No critical data quality issues found.")

        lines.append("============================================================")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "class_counts": self.class_counts,
            "imbalance_ratio": round(self.imbalance_ratio, 2),
            "corrupt_file_count": len(self.corrupt_files),
            "corrupt_files": [
                {"file_path": c.file_path, "class": c.class_name, "error": c.error}
                for c in self.corrupt_files
            ],
            "dimension_stats": self.dimension_stats,
            "duplicate_cluster_count": len(self.duplicate_clusters),
            "redundant_duplicates_count": self.total_duplicates_removed_count,
            "duplicate_clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "category": c.category,
                    "survivor_id": str(c.survivor_id),
                    "duplicate_ids": [str(d) for d in c.duplicate_ids],
                    "total_count": c.total_count,
                }
                for c in self.duplicate_clusters
            ],
            "cross_label_conflict_count": len(self.cross_label_conflicts),
            "leakage_count": len(self.leakage_matches),
            "warnings": self.warnings,
        }

    def get_clean_indices(
        self,
        remove_duplicates: bool = True,
        remove_cross_label: bool = True,
    ) -> List[int]:
        """Return clean integer indices for torch.utils.data.Subset.

        Excludes duplicate redundant copies (keeping only the best survivor)
        and optionally removes cross-label ambiguous conflicts.
        """
        exclude_ids: Set[Any] = set()
        if remove_duplicates:
            for cluster in self.duplicate_clusters:
                exclude_ids.update(cluster.duplicate_ids)
        if remove_cross_label:
            for conf in self.cross_label_conflicts:
                # Remove both or later sample
                exclude_ids.add(conf.sample_b_id)

        clean_indices: List[int] = []
        for s in self.samples:
            if s.sample_id not in exclude_ids:
                if isinstance(s.sample_id, int):
                    clean_indices.append(s.sample_id)
                elif isinstance(s.sample_id, str) and s.sample_id.isdigit():
                    clean_indices.append(int(s.sample_id))
        return clean_indices

    def export_clean_dataset(
        self,
        output_dir: str,
        mode: str = "copy",
        remove_duplicates: bool = True,
        remove_cross_label: bool = True,
    ) -> int:
        """Export clean files to output_dir mirroring class subfolder structure.

        Returns number of clean files written.
        """
        exclude_paths: Set[str] = set()
        if remove_duplicates:
            for cluster in self.duplicate_clusters:
                for dup_id in cluster.duplicate_ids:
                    if isinstance(dup_id, str) and os.path.exists(dup_id):
                        exclude_paths.add(os.path.normpath(dup_id))
        if remove_cross_label:
            for conf in self.cross_label_conflicts:
                if isinstance(conf.sample_b_id, str) and os.path.exists(conf.sample_b_id):
                    exclude_paths.add(os.path.normpath(conf.sample_b_id))

        os.makedirs(output_dir, exist_ok=True)
        copied = 0
        for s in self.samples:
            if not s.file_path or not os.path.exists(s.file_path):
                continue
            if os.path.normpath(s.file_path) in exclude_paths:
                continue

            target_class_dir = os.path.join(output_dir, s.class_name)
            os.makedirs(target_class_dir, exist_ok=True)
            dst_file = os.path.join(target_class_dir, os.path.basename(s.file_path))

            # Handle name collision if any
            if os.path.exists(dst_file):
                base, ext = os.path.splitext(os.path.basename(s.file_path))
                dst_file = os.path.join(target_class_dir, f"{base}_{copied}{ext}")

            if mode == "copy":
                shutil.copy2(s.file_path, dst_file)
            elif mode == "move":
                shutil.move(s.file_path, dst_file)
            copied += 1

        return copied

    def export_manifest(self, output_path: str) -> None:
        """Export JSON manifest describing each audited file and its audit status."""
        exclude_duplicates = {d: c.cluster_id for c in self.duplicate_clusters for d in c.duplicate_ids}
        survivors = {c.survivor_id: c.cluster_id for c in self.duplicate_clusters}
        cross_conflicts = {c.sample_b_id: f"{c.class_a}->{c.class_b}" for c in self.cross_label_conflicts}

        records = []
        for s in self.samples:
            status = "clean"
            note = ""
            if s.sample_id in exclude_duplicates:
                status = "duplicate_removed"
                note = f"Belongs to cluster {exclude_duplicates[s.sample_id]}"
            elif s.sample_id in survivors:
                status = "duplicate_survivor"
                note = f"Kept survivor for cluster {survivors[s.sample_id]}"
            elif s.sample_id in cross_conflicts:
                status = "cross_label_conflict"
                note = f"Contradiction: {cross_conflicts[s.sample_id]}"

            records.append({
                "sample_id": str(s.sample_id),
                "class_name": s.class_name,
                "file_path": s.file_path,
                "width": s.width,
                "height": s.height,
                "sharpness": round(s.sharpness, 2),
                "status": status,
                "note": note,
            })

        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def save_html(self, output_path: str, title: str = "Lensight Dataset Health & Audit Report") -> str:
        """Generate and save interactive HTML audit dashboard."""
        from .eda_report import build_audit_html_report
        html_content = build_audit_html_report(self, title=title)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path

    def _repr_html_(self) -> str:
        """Inline notebook renderer for Jupyter, Colab, and VS Code."""
        from .eda_report import build_audit_html_report
        return build_audit_html_report(self)


class DatasetAuditor:
    """Auditor engine for dataset health, leakage, duplicates, and cross-label conflicts."""

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        max_samples: Optional[int] = None,
        store_thumbnails: bool = True,
        max_thumbnails: int = 100,
        random_seed: int = 42,
    ):
        self.class_names = list(class_names) if class_names is not None else None
        self.max_samples = max_samples
        self.store_thumbnails = store_thumbnails
        self.max_thumbnails = max_thumbnails
        self.random_seed = random_seed

    def _scan_directory(
        self,
        root_dir: str,
        split_name: str = "dataset",
        algo: str = "dhash",
    ) -> Tuple[List[SampleMeta], List[CorruptFileInfo]]:
        """Recursively scan class folders in root_dir."""
        subdirs = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        if not subdirs:
            raise ValueError(
                f"No class subfolders found in {root_dir}. Expected folder structure: root_dir/<class_name>/<images>"
            )

        samples: List[SampleMeta] = []
        corrupt: List[CorruptFileInfo] = []
        thumb_count = 0

        for c_idx, c_name in enumerate(subdirs):
            cls_dir = os.path.join(root_dir, c_name)
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                file_path = os.path.join(cls_dir, fname)
                try:
                    with Image.open(file_path) as im:
                        im.load()
                        w, h = im.size
                        channels = len(im.getbands())
                        gray = np.asarray(im.convert("L"), dtype=np.float32)
                        sharpness = _calc_laplacian_sharpness(gray)
                        mean_val = float(gray.mean() / 255.0)

                    # Compute perceptual hash
                    hval = compute_hash(file_path, algo=algo)

                    thumb = None
                    if self.store_thumbnails and thumb_count < self.max_thumbnails:
                        thumb = _image_to_base64_thumb(file_path)
                        thumb_count += 1

                    samples.append(
                        SampleMeta(
                            sample_id=file_path,
                            class_name=c_name,
                            class_idx=c_idx,
                            split=split_name,
                            hash_val=hval,
                            width=w,
                            height=h,
                            channels=channels,
                            sharpness=sharpness,
                            mean_intensity=mean_val,
                            file_path=file_path,
                            thumb_b64=thumb,
                        )
                    )
                except Exception as e:
                    corrupt.append(
                        CorruptFileInfo(
                            file_path=file_path,
                            class_name=c_name,
                            error=str(e),
                        )
                    )

                if self.max_samples and len(samples) >= self.max_samples:
                    break
            if self.max_samples and len(samples) >= self.max_samples:
                break

        return samples, corrupt

    def _scan_iterable(
        self,
        data_source: Any,
        split_name: str = "dataset",
        algo: str = "dhash",
    ) -> List[SampleMeta]:
        """Scan PyTorch DataLoader, Dataset, or list."""
        samples: List[SampleMeta] = []
        sample_idx = 0
        thumb_count = 0

        # Handle PyTorch DataLoader
        if hasattr(data_source, "dataset") and hasattr(data_source, "__iter__"):
            for batch in data_source:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    imgs, labels = batch[0], batch[1]
                else:
                    imgs = batch
                    labels = [0] * len(imgs)

                batch_len = len(imgs)
                for b in range(batch_len):
                    img = imgs[b]
                    lbl = labels[b]
                    if hasattr(lbl, "item"):
                        lbl = lbl.item()
                    c_idx = int(lbl)
                    c_name = (
                        self.class_names[c_idx]
                        if (self.class_names and c_idx < len(self.class_names))
                        else f"class_{c_idx}"
                    )

                    # Extract dimensions and sharpness
                    if hasattr(img, "cpu"):
                        arr = img.detach().cpu().numpy()
                        if arr.ndim == 3 and arr.shape[0] in (1, 3):
                            arr = np.transpose(arr, (1, 2, 0))
                        h, w = arr.shape[0], arr.shape[1]
                        ch = arr.shape[2] if arr.ndim == 3 else 1
                        gray = arr.mean(axis=2) if arr.ndim == 3 else arr
                        if gray.max() <= 1.05 and gray.min() >= -0.05:
                            gray = gray * 255.0
                        sharpness = _calc_laplacian_sharpness(gray.astype(np.float32))
                        mean_val = float(gray.mean() / 255.0)
                    elif isinstance(img, Image.Image):
                        w, h = img.size
                        ch = len(img.getbands())
                        gray = np.asarray(img.convert("L"), dtype=np.float32)
                        sharpness = _calc_laplacian_sharpness(gray)
                        mean_val = float(gray.mean() / 255.0)
                    else:
                        w, h, ch, sharpness, mean_val = 64, 64, 3, 0.0, 0.5

                    hval = compute_hash(img, algo=algo)
                    thumb = None
                    if self.store_thumbnails and thumb_count < self.max_thumbnails:
                        thumb = _image_to_base64_thumb(img)
                        thumb_count += 1

                    samples.append(
                        SampleMeta(
                            sample_id=sample_idx,
                            class_name=c_name,
                            class_idx=c_idx,
                            split=split_name,
                            hash_val=hval,
                            width=w,
                            height=h,
                            channels=ch,
                            sharpness=sharpness,
                            mean_intensity=mean_val,
                            thumb_b64=thumb,
                        )
                    )
                    sample_idx += 1
                    if self.max_samples and len(samples) >= self.max_samples:
                        return samples
        else:
            # Handle Dataset or sequence
            n = len(data_source)
            limit = min(n, self.max_samples) if self.max_samples else n
            for i in range(limit):
                item = data_source[i]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    img, lbl = item[0], item[1]
                else:
                    img, lbl = item, 0
                if hasattr(lbl, "item"):
                    lbl = lbl.item()
                c_idx = int(lbl)
                c_name = (
                    self.class_names[c_idx]
                    if (self.class_names and c_idx < len(self.class_names))
                    else f"class_{c_idx}"
                )

                if hasattr(img, "cpu"):
                    arr = img.detach().cpu().numpy()
                    if arr.ndim == 3 and arr.shape[0] in (1, 3):
                        arr = np.transpose(arr, (1, 2, 0))
                    h, w = arr.shape[0], arr.shape[1]
                    ch = arr.shape[2] if arr.ndim == 3 else 1
                    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
                    if gray.max() <= 1.05 and gray.min() >= -0.05:
                        gray = gray * 255.0
                    sharpness = _calc_laplacian_sharpness(gray.astype(np.float32))
                    mean_val = float(gray.mean() / 255.0)
                elif isinstance(img, Image.Image):
                    w, h = img.size
                    ch = len(img.getbands())
                    gray = np.asarray(img.convert("L"), dtype=np.float32)
                    sharpness = _calc_laplacian_sharpness(gray)
                    mean_val = float(gray.mean() / 255.0)
                else:
                    w, h, ch, sharpness, mean_val = 64, 64, 3, 0.0, 0.5

                hval = compute_hash(img, algo=algo)
                thumb = None
                if self.store_thumbnails and thumb_count < self.max_thumbnails:
                    thumb = _image_to_base64_thumb(img)
                    thumb_count += 1

                samples.append(
                    SampleMeta(
                        sample_id=i,
                        class_name=c_name,
                        class_idx=c_idx,
                        split=split_name,
                        hash_val=hval,
                        width=w,
                        height=h,
                        channels=ch,
                        sharpness=sharpness,
                        mean_intensity=mean_val,
                        thumb_b64=thumb,
                    )
                )

        return samples

    def audit(
        self,
        data_source: Any,
        threshold: int = 3,
        algo: str = "dhash",
        detect_cross_label: bool = True,
    ) -> AuditReport:
        """Run a full comprehensive health and duplication audit on a dataset.

        Args:
            data_source: Directory path string, PyTorch DataLoader, Dataset, or list.
            threshold: Hamming distance threshold (default: 3).
            algo: Perceptual hash algorithm ('dhash', 'ahash', 'phash').
            detect_cross_label: Whether to scan for conflicting cross-label duplicates.

        Returns:
            AuditReport with clusters, cross-label conflicts, stats, and warnings.
        """
        corrupt: List[CorruptFileInfo] = []
        if isinstance(data_source, str) and os.path.isdir(data_source):
            samples, corrupt = self._scan_directory(data_source, algo=algo)
        else:
            samples = self._scan_iterable(data_source, algo=algo)

        if not samples:
            raise ValueError("No valid samples could be read from data_source.")

        # 1. Class Distribution & Imbalance
        class_counts: Dict[str, int] = {}
        for s in samples:
            class_counts[s.class_name] = class_counts.get(s.class_name, 0) + 1

        counts_list = list(class_counts.values())
        imbalance_ratio = float(max(counts_list) / min(counts_list)) if counts_list else 1.0

        # 2. Dimensions stats
        widths = [s.width for s in samples]
        heights = [s.height for s in samples]
        unique_shapes = len(set(zip(widths, heights)))
        dim_stats = {
            "width_min": min(widths),
            "width_max": max(widths),
            "height_min": min(heights),
            "height_max": max(heights),
            "unique_resolutions": unique_shapes,
            "has_varying_resolutions": unique_shapes > 1,
        }

        # 3. Transitive Duplicate Clustering (with Smart Survivor Selection)
        items = [s.sample_id for s in samples]
        hashes = [s.hash_val for s in samples]
        qualities = [s.sharpness for s in samples]
        id_to_sample = {s.sample_id: s for s in samples}

        # Cluster within classes or globally
        clusters = cluster_transitive_duplicates(
            items=items,
            hashes=hashes,
            threshold=threshold,
            quality_scores=qualities,
        )

        # 4. Cross-Label Conflicts Detection
        cross_conflicts: List[CrossLabelConflict] = []
        if detect_cross_label and len(class_counts) > 1:
            # Group samples by class
            samples_by_class: Dict[str, List[SampleMeta]] = {}
            for s in samples:
                samples_by_class.setdefault(s.class_name, []).append(s)

            class_keys = sorted(samples_by_class.keys())
            for i in range(len(class_keys)):
                cls_a = class_keys[i]
                list_a = samples_by_class[cls_a]
                hashes_a = [s.hash_val for s in list_a]

                for j in range(i + 1, len(class_keys)):
                    cls_b = class_keys[j]
                    list_b = samples_by_class[cls_b]
                    hashes_b = [s.hash_val for s in list_b]

                    # Vectorized pairwise Hamming matrix
                    d_mat = pairwise_hamming_matrix(hashes_a, hashes_b)
                    matched_coords = np.argwhere(d_mat <= threshold)

                    for idx_a, idx_b in matched_coords:
                        sa = list_a[idx_a]
                        sb = list_b[idx_b]
                        dist = int(d_mat[idx_a, idx_b])
                        cross_conflicts.append(
                            CrossLabelConflict(
                                sample_a_id=sa.sample_id,
                                class_a=cls_a,
                                sample_b_id=sb.sample_id,
                                class_b=cls_b,
                                hamming_distance=dist,
                                is_exact=bool(dist == 0),
                                thumb_a=sa.thumb_b64,
                                thumb_b=sb.thumb_b64,
                            )
                        )

        # 5. Build Actionable Warnings
        warnings: List[str] = []
        if imbalance_ratio >= 3.0:
            warnings.append(
                f"Severe class imbalance: max-to-min ratio is {imbalance_ratio:.2f}x. Long-tail classes risk being ignored."
            )
        if corrupt:
            warnings.append(f"Found {len(corrupt)} corrupt or unreadable image file(s).")
        if clusters:
            tot_dup = sum(len(c.duplicate_ids) for c in clusters)
            warnings.append(
                f"Found {len(clusters)} duplicate clusters containing {tot_dup} redundant image(s)."
            )
        if cross_conflicts:
            warnings.append(
                f"CRITICAL: Found {len(cross_conflicts)} cross-label duplicate conflict(s). Same image exists with conflicting labels!"
            )
        if dim_stats["has_varying_resolutions"]:
            warnings.append(
                f"Dataset contains {unique_shapes} varying resolutions. Ensure resize transforms do not distort aspect ratios."
            )

        return AuditReport(
            total_samples=len(samples),
            class_counts=class_counts,
            imbalance_ratio=imbalance_ratio,
            corrupt_files=corrupt,
            dimension_stats=dim_stats,
            duplicate_clusters=clusters,
            cross_label_conflicts=cross_conflicts,
            samples=samples,
            warnings=warnings,
            threshold=threshold,
            algo=algo,
        )

    def check_leakage(
        self,
        train_source: Any,
        test_source: Any,
        threshold: int = 3,
        algo: str = "dhash",
    ) -> LeakageReport:
        """Check for train/test data leakage.

        Compares test samples against train samples. If any test sample matches
        a train sample within threshold, it is flagged as leaked.
        """
        # Scan train
        if isinstance(train_source, str) and os.path.isdir(train_source):
            train_samples, _ = self._scan_directory(train_source, split_name="train", algo=algo)
        else:
            train_samples = self._scan_iterable(train_source, split_name="train", algo=algo)

        # Scan test
        if isinstance(test_source, str) and os.path.isdir(test_source):
            test_samples, _ = self._scan_directory(test_source, split_name="test", algo=algo)
        else:
            test_samples = self._scan_iterable(test_source, split_name="test", algo=algo)

        if not train_samples or not test_samples:
            raise ValueError("Train or test source has no readable samples.")

        # Group train samples by class for fast class-aligned comparison
        train_by_class: Dict[str, List[SampleMeta]] = {}
        for s in train_samples:
            train_by_class.setdefault(s.class_name, []).append(s)

        test_by_class: Dict[str, List[SampleMeta]] = {}
        for s in test_samples:
            test_by_class.setdefault(s.class_name, []).append(s)

        all_matches: List[LeakageMatch] = []
        per_class_stats: Dict[str, Dict[str, Any]] = {}

        common_classes = sorted(set(train_by_class.keys()) | set(test_by_class.keys()))

        for cls in common_classes:
            tr_list = train_by_class.get(cls, [])
            te_list = test_by_class.get(cls, [])

            cls_leaked = 0
            if tr_list and te_list:
                tr_hashes = [s.hash_val for s in tr_list]
                te_hashes = [s.hash_val for s in te_list]

                # Compute pairwise Hamming matrix
                dist_mat = pairwise_hamming_matrix(te_hashes, tr_hashes)  # shape (N_test, N_train)
                min_dists = dist_mat.min(axis=1)  # closest train match for each test sample
                best_train_indices = dist_mat.argmin(axis=1)

                for test_idx, min_dist in enumerate(min_dists):
                    if min_dist <= threshold:
                        cls_leaked += 1
                        tr_match = tr_list[best_train_indices[test_idx]]
                        te_sample = te_list[test_idx]
                        all_matches.append(
                            LeakageMatch(
                                test_id=te_sample.sample_id,
                                test_class=cls,
                                train_id=tr_match.sample_id,
                                train_class=cls,
                                hamming_distance=int(min_dist),
                                is_exact=bool(min_dist == 0),
                                test_thumb=te_sample.thumb_b64,
                                train_thumb=tr_match.thumb_b64,
                            )
                        )

            per_class_stats[cls] = {
                "train_count": len(tr_list),
                "test_count": len(te_list),
                "leaked": cls_leaked,
                "leakage_rate": (cls_leaked / len(te_list)) if te_list else 0.0,
            }

        return LeakageReport(
            train_count=len(train_samples),
            test_count=len(test_samples),
            leaked_count=len(all_matches),
            per_class_leakage=per_class_stats,
            matches=all_matches,
            threshold=threshold,
            algo=algo,
        )


# Functional convenience helpers
def audit_dataset(
    data_source: Any,
    class_names: Optional[Sequence[str]] = None,
    threshold: int = 3,
    algo: str = "dhash",
) -> AuditReport:
    """Convenience function: audit a dataset for duplicates, conflicts, and health anomalies."""
    auditor = DatasetAuditor(class_names=class_names)
    return auditor.audit(data_source, threshold=threshold, algo=algo)


def check_leakage(
    train_source: Any,
    test_source: Any,
    class_names: Optional[Sequence[str]] = None,
    threshold: int = 3,
    algo: str = "dhash",
) -> LeakageReport:
    """Convenience function: check for train/test data leakage."""
    auditor = DatasetAuditor(class_names=class_names)
    return auditor.check_leakage(train_source, test_source, threshold=threshold, algo=algo)


def check_cross_label(
    data_source: Any,
    class_names: Optional[Sequence[str]] = None,
    threshold: int = 3,
    algo: str = "dhash",
) -> CrossLabelReport:
    """Convenience function: detect cross-label contradictory duplicate images."""
    report = audit_dataset(data_source, class_names=class_names, threshold=threshold, algo=algo)
    pair_counts: Dict[Tuple[str, str], int] = {}
    for c in report.cross_label_conflicts:
        k = (c.class_a, c.class_b) if c.class_a < c.class_b else (c.class_b, c.class_a)
        pair_counts[k] = pair_counts.get(k, 0) + 1

    return CrossLabelReport(
        total_samples=report.total_samples,
        num_classes=len(report.class_counts),
        conflict_count=len(report.cross_label_conflicts),
        conflicts=report.cross_label_conflicts,
        conflicted_class_pairs=pair_counts,
        threshold=threshold,
        algo=algo,
    )
