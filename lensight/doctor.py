"""
ModelDoctor — the one-call entry point.

Wraps GradCAM + MisclassificationAnalyzer + ConfidenceAnalyzer + the HTML
report builder into a single `diagnose(dataloader)` call, so debugging a
freshly trained CV classifier doesn't require wiring four modules together
by hand every time.
"""

from __future__ import annotations
import html
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .cam.gradcam import GradCAM
from .analysis.misclassification import MisclassificationAnalyzer, MisclassificationReport
from .analysis.confidence import ConfidenceAnalyzer, CalibrationReport
from .analysis.temperature_scaling import TemperatureScaler
from .report.html_report import build_html_report
from .utils.image_utils import denormalize, overlay_heatmap
from .utils.model_utils import find_last_linear_layer


@dataclass
class DiagnosisReport:
    misclassification: MisclassificationReport
    calibration: CalibrationReport
    html: str
    eda: Any | None = None

    def save_html(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html)

    def summary_text(self) -> str:
        text = (
            self.misclassification.summary_text()
            + "\n\n"
            + self.calibration.summary_text()
        )
        if self.eda is not None:
            text += "\n\n" + self.eda.summary_text()
        return text

    def to_dict(self) -> dict:
        d = {
            "misclassification": self.misclassification.to_dict(),
            "calibration": self.calibration.to_dict(),
        }
        if self.eda is not None:
            d["eda"] = self.eda.to_dict()
        return d

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        json_str = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def _repr_html_(self) -> str:
        """Render interactive dashboard directly in Jupyter, VS Code, or Colab notebooks."""
        escaped_html = html.escape(self.html)
        return (
            f'<iframe style="width: 100%; height: 750px; border: 1px solid #cbd5e1; '
            f'border-radius: 8px;" srcdoc="{escaped_html}"></iframe>'
        )


class ModelDoctor:
    """
    Usage
    -----
        doctor = ModelDoctor(model, class_names=classes)
        report = doctor.diagnose(val_dataloader)
        print(report.summary_text())
        report.save_html("diagnosis.html")
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: list[str] | None = None,
        num_classes: int | None = None,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
    ):
        self.model = model
        self.model.eval()
        self.class_names = class_names
        if num_classes is None:
            num_classes = find_last_linear_layer(model).out_features
        self.num_classes = num_classes
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def diagnose(
        self,
        dataloader,
        n_clusters: int = 6,
        n_example_images: int = 6,
        cam_target_layer: nn.Module | None = None,
        report_title: str = "Model Diagnosis Report",
        compute_calibration_remediation: bool = True,
        include_eda: bool = False,
        max_eda_samples: int = 500,
    ) -> DiagnosisReport:
        misclass_analyzer = MisclassificationAnalyzer(
            self.model, num_classes=self.num_classes, class_names=self.class_names
        )
        misclass_report = misclass_analyzer.analyze(dataloader, n_clusters=n_clusters)

        confidence_analyzer = ConfidenceAnalyzer(self.model)
        calibration_report = confidence_analyzer.analyze(dataloader)

        if compute_calibration_remediation:
            try:
                scaler = TemperatureScaler(self.model)
                cal_summary = scaler.fit(dataloader)
                calibration_report.suggested_temperature = cal_summary.temperature
                calibration_report.calibrated_ece = cal_summary.calibrated_ece
            except Exception:
                pass

        eda_report = None
        if include_eda:
            try:
                from .eda.explorer import DatasetExplorer
                explorer = DatasetExplorer(dataloader, class_names=self.class_names)
                eda_report = explorer.analyze(max_samples=max_eda_samples)
            except Exception:
                pass

        example_images = self._render_example_explanations(
            misclass_report, cam_target_layer, n_example_images
        )

        html_content = build_html_report(
            misclass_report=misclass_report,
            calibration_report=calibration_report,
            example_images=example_images,
            title=report_title,
            eda_report=eda_report,
        )

        return DiagnosisReport(
            misclassification=misclass_report,
            calibration=calibration_report,
            html=html_content,
            eda=eda_report,
        )

    def _render_example_explanations(self, misclass_report, cam_target_layer, n_example_images):
        if not misclass_report.samples or n_example_images <= 0:
            return []

        cam = GradCAM(self.model, target_layer=cam_target_layer)

        # Prioritize suspected label errors and distinct failure clusters
        label_error_samples = misclass_report.find_label_errors(0.80)
        label_error_indices = {s.index for s in label_error_samples}
        picks = []

        # Include up to 2 high-confidence label errors first if available
        if label_error_samples:
            sorted_errs = sorted(label_error_samples, key=lambda s: -s.confidence)
            picks.extend(sorted_errs[:2])

        pick_indices = {s.index for s in picks}

        # Spread remaining picks across clusters
        samples = misclass_report.samples
        if misclass_report.n_clusters > 0:
            by_cluster: dict[int, list] = {}
            for s in samples:
                if s.index not in pick_indices:
                    by_cluster.setdefault(s.cluster, []).append(s)
            clusters = sorted(by_cluster.keys())
            i = 0
            while len(picks) < min(n_example_images, len(samples)) and clusters:
                c = clusters[i % len(clusters)]
                bucket = by_cluster[c]
                if bucket:
                    p = bucket.pop(0)
                    picks.append(p)
                    pick_indices.add(p.index)
                else:
                    clusters.remove(c)
                i += 1
        else:
            for s in samples:
                if s.index not in pick_indices and len(picks) < n_example_images:
                    picks.append(s)
                    pick_indices.add(s.index)

        images = []
        for sample in picks:
            if sample.image.numel() == 0:
                continue
            heatmap = cam.explain(sample.image, target_class=sample.pred_label)
            rgb = denormalize(sample.image, mean=self.normalize_mean, std=self.normalize_std)
            overlay = overlay_heatmap(rgb, heatmap)
            is_label_err = sample.index in label_error_indices
            prefix = "[Label Error Candidate] " if is_label_err else ""
            caption = (
                f"{prefix}true: {misclass_report.label_name(sample.true_label)} | "
                f"pred: {misclass_report.label_name(sample.pred_label)} "
                f"({sample.confidence:.0%})"
            )
            meta = {
                "cluster": sample.cluster,
                "is_label_error": is_label_err,
                "true_label": sample.true_label,
                "pred_label": sample.pred_label,
            }
            images.append((caption, overlay, meta))
        return images
