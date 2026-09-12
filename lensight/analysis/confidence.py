"""
Confidence calibration analysis.

A model can be accurate and still badly calibrated — saying "99% confident"
on things it gets wrong 20% of the time. That gap matters anywhere a
downstream system trusts the confidence score itself (auto-accept /
auto-flag-for-review thresholds, active learning, ensembling). This module
computes Expected Calibration Error (ECE) and the underlying reliability
diagram bins so the gap is visible and quantified rather than assumed away.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..utils.model_utils import get_device


@dataclass
class CalibrationReport:
    n_bins: int
    bin_confidence: np.ndarray  # avg predicted confidence per bin
    bin_accuracy: np.ndarray  # avg empirical accuracy per bin
    bin_counts: np.ndarray  # number of samples per bin
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    overall_accuracy: float
    overall_confidence: float
    suggested_temperature: float | None = None
    calibrated_ece: float | None = None

    def summary_text(self) -> str:
        verdict = (
            "well-calibrated"
            if self.ece < 0.03
            else "mildly overconfident/underconfident"
            if self.ece < 0.10
            else "poorly calibrated"
        )
        base = (
            f"ECE: {self.ece:.4f}  |  MCE: {self.mce:.4f}  |  "
            f"mean confidence {self.overall_confidence:.2%} vs "
            f"accuracy {self.overall_accuracy:.2%}  -> {verdict}"
        )
        if self.suggested_temperature is not None and self.calibrated_ece is not None:
            base += f"\n  [Remediation] Temperature scaling (T={self.suggested_temperature:.3f}) reduces ECE to {self.calibrated_ece:.4f}"
        return base

    def to_dict(self) -> dict:
        d = {
            "n_bins": self.n_bins,
            "ece": float(self.ece),
            "mce": float(self.mce),
            "overall_accuracy": float(self.overall_accuracy),
            "overall_confidence": float(self.overall_confidence),
            "bin_confidence": self.bin_confidence.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "bin_counts": self.bin_counts.tolist(),
        }
        if self.suggested_temperature is not None:
            d["suggested_temperature"] = self.suggested_temperature
        if self.calibrated_ece is not None:
            d["calibrated_ece"] = self.calibrated_ece
        return d


class ConfidenceAnalyzer:
    """
    Usage
    -----
        analyzer = ConfidenceAnalyzer(model)
        report = analyzer.analyze(dataloader, n_bins=15)
        print(report.summary_text())
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.device = get_device(model)

    def analyze(self, dataloader, n_bins: int = 15) -> CalibrationReport:
        all_confidences = []
        all_correct = []

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device)
                labels = labels.to(self.device)

                probs = torch.softmax(self.model(images), dim=1)
                confs, preds = probs.max(dim=1)

                all_confidences.append(confs.cpu().numpy())
                all_correct.append((preds == labels).cpu().numpy())

        confidences = np.concatenate(all_confidences)
        correct = np.concatenate(all_correct).astype(float)

        ece, mce, bin_confidence, bin_accuracy, bin_counts = compute_calibration_bins(
            confidences, correct, n_bins=n_bins
        )

        return CalibrationReport(
            n_bins=n_bins,
            bin_confidence=bin_confidence,
            bin_accuracy=bin_accuracy,
            bin_counts=bin_counts,
            ece=float(ece),
            mce=float(mce),
            overall_accuracy=float(correct.mean()),
            overall_confidence=float(confidences.mean()),
        )


def compute_calibration_bins(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (ece, mce, bin_confidence, bin_accuracy, bin_counts)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidence = np.zeros(n_bins)
    bin_accuracy = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    ece = 0.0
    mce = 0.0
    n = len(confidences)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        bin_counts[b] = count
        if count > 0:
            bin_confidence[b] = float(confidences[mask].mean())
            bin_accuracy[b] = float(correct[mask].mean())
            gap = abs(bin_confidence[b] - bin_accuracy[b])
            ece += (count / n) * gap
            mce = max(mce, gap)

    return float(ece), float(mce), bin_confidence, bin_accuracy, bin_counts


class ExpectedCalibrationError:
    """Standalone calculator for Expected Calibration Error (ECE)."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def compute(self, probs: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        correct = (preds == labels).astype(float)
        ece, _, _, _, _ = compute_calibration_bins(confs, correct, self.n_bins)
        return ece

    def __call__(self, probs: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
        return self.compute(probs, labels)


class MaximumCalibrationError:
    """Standalone calculator for Maximum Calibration Error (MCE)."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def compute(self, probs: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        correct = (preds == labels).astype(float)
        _, mce, _, _, _ = compute_calibration_bins(confs, correct, self.n_bins)
        return mce

    def __call__(self, probs: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
        return self.compute(probs, labels)

