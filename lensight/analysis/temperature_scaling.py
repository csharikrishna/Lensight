"""
Temperature Scaling for post-hoc probability calibration.

Reference:
    Guo, Pleiss, Sun, & Weinberger (ICML 2017).
    "On Calibration of Modern Neural Networks"
    https://arxiv.org/abs/1706.04599

Modern deep neural networks often output overconfident probability estimates.
Temperature scaling introduces a single learned scalar parameter T > 0 that
scales unnormalized logits before the softmax activation:
    p_i = exp(z_i / T) / sum_j exp(z_j / T)

Because T is strictly positive and monotonic, it preserves the original top-1
classification accuracy while significantly lowering Expected Calibration Error (ECE).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.model_utils import get_device


def _calculate_ece_and_mce(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float]:
    """Calculate ECE and MCE from confidence array and binary correctness array."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0, 0.0

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        if count > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correct[mask].mean()
            gap = abs(bin_conf - bin_acc)
            ece += (count / n) * gap
            mce = max(mce, gap)

    return float(ece), float(mce)


@dataclass
class CalibrationSummary:
    temperature: float
    initial_ece: float
    calibrated_ece: float
    initial_mce: float
    calibrated_mce: float
    initial_nll: float
    calibrated_nll: float

    @property
    def ece_reduction(self) -> float:
        return max(0.0, self.initial_ece - self.calibrated_ece)

    @property
    def ece_improvement_pct(self) -> float:
        if self.initial_ece <= 1e-8:
            return 0.0
        return (self.ece_reduction / self.initial_ece) * 100.0

    def summary_text(self) -> str:
        return (
            f"Temperature: {self.temperature:.3f}\n"
            f"ECE: {self.initial_ece:.4f} -> {self.calibrated_ece:.4f} "
            f"(-{self.ece_improvement_pct:.1f}% error)\n"
            f"MCE: {self.initial_mce:.4f} -> {self.calibrated_mce:.4f}\n"
            f"NLL: {self.initial_nll:.4f} -> {self.calibrated_nll:.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "temperature": self.temperature,
            "initial_ece": self.initial_ece,
            "calibrated_ece": self.calibrated_ece,
            "ece_reduction": self.ece_reduction,
            "ece_improvement_pct": self.ece_improvement_pct,
            "initial_mce": self.initial_mce,
            "calibrated_mce": self.calibrated_mce,
            "initial_nll": self.initial_nll,
            "calibrated_nll": self.calibrated_nll,
        }


class CalibratedModel(nn.Module):
    """
    Wrapper around an existing classifier model that scales logits by a learned
    temperature parameter T before returning them.

    Usage
    -----
        calibrated_model = scaler.calibrated_model
        logits = calibrated_model(images)
        probs = calibrated_model.predict_proba(images)
    """

    def __init__(self, model: nn.Module, temperature: float | torch.Tensor):
        super().__init__()
        self.model = model
        temp_val = float(temperature.item()) if isinstance(temperature, torch.Tensor) else float(temperature)
        self.register_buffer("temperature", torch.tensor(max(1e-4, temp_val), dtype=torch.float32))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        logits = self.model(*args, **kwargs)
        return logits / self.temperature

    def predict_proba(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return softmax probabilities using calibrated logits."""
        logits = self.forward(*args, **kwargs)
        return F.softmax(logits, dim=-1)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class TemperatureScaler:
    """
    Learns the optimal scalar temperature T on validation logits to minimize
    Negative Log Likelihood (NLL) and calibrate probabilities.

    Usage
    -----
        scaler = TemperatureScaler(model)
        summary = scaler.fit(val_dataloader)
        print(summary.summary_text())
        calibrated_model = scaler.calibrated_model
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.device = get_device(model)
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)

    def fit(
        self,
        dataloader,
        lr: float = 0.01,
        max_iter: int = 50,
        n_bins: int = 15,
    ) -> CalibrationSummary:
        """
        Fit temperature parameter T using L-BFGS on validation logits.

        Parameters
        ----------
        dataloader : DataLoader yielding (images, labels) or (images, labels, ...)
        lr : learning rate for L-BFGS optimizer
        max_iter : maximum optimization iterations
        n_bins : number of bins for ECE/MCE computation

        Returns
        -------
        CalibrationSummary with before/after calibration metrics.
        """
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device)
                labels = labels.to(self.device)
                out = self.model(images)
                logits_list.append(out)
                labels_list.append(labels)

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        nll_criterion = nn.CrossEntropyLoss()

        # Compute initial metrics
        with torch.no_grad():
            init_nll = float(nll_criterion(logits, labels).item())
            init_probs = F.softmax(logits, dim=1)
            init_confs, init_preds = init_probs.max(dim=1)
            init_correct = (init_preds == labels).cpu().numpy().astype(float)
            init_ece, init_mce = _calculate_ece_and_mce(
                init_confs.cpu().numpy(), init_correct, n_bins=n_bins
            )

        # Optimize temperature with L-BFGS
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            # Ensure temperature stays positive
            t = torch.clamp(self.temperature, min=1e-3)
            loss = nll_criterion(logits / t, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)
            cal_nll = float(nll_criterion(logits / self.temperature, labels).item())
            cal_probs = F.softmax(logits / self.temperature, dim=1)
            cal_confs, cal_preds = cal_probs.max(dim=1)
            cal_correct = (cal_preds == labels).cpu().numpy().astype(float)
            cal_ece, cal_mce = _calculate_ece_and_mce(
                cal_confs.cpu().numpy(), cal_correct, n_bins=n_bins
            )

        return CalibrationSummary(
            temperature=float(self.temperature.item()),
            initial_ece=init_ece,
            calibrated_ece=cal_ece,
            initial_mce=init_mce,
            calibrated_mce=cal_mce,
            initial_nll=init_nll,
            calibrated_nll=cal_nll,
        )

    @property
    def calibrated_model(self) -> CalibratedModel:
        """Return the wrapped model configured with the fitted temperature."""
        return CalibratedModel(self.model, self.temperature)
