"""Integrated Gradients (Sundararajan et al., 2017, https://arxiv.org/abs/1703.01365).

Attributes the prediction to input pixels by accumulating gradients along a
straight-line path from a baseline (typically a black image) to the actual
input. Satisfies useful axioms (sensitivity, implementation invariance) that
vanilla gradients don't, at the cost of `steps` extra forward/backward passes.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ..utils.model_utils import get_device


class IntegratedGradients:
    def __init__(self, model: nn.Module, steps: int = 50):
        self.model = model
        self.model.eval()
        self.steps = steps
        self.device = get_device(model)

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        baseline: torch.Tensor | None = None,
    ) -> np.ndarray:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        baseline = baseline.to(self.device)

        if target_class is None:
            with torch.no_grad():
                target_class = int(self.model(input_tensor).argmax(dim=1).item())

        alphas = torch.linspace(0, 1, self.steps, device=self.device)
        total_grads = torch.zeros_like(input_tensor)

        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.clone().requires_grad_(True)

            scores = self.model(interpolated)
            self.model.zero_grad(set_to_none=True)
            scores[0, target_class].backward()

            total_grads += interpolated.grad.detach()

        avg_grads = total_grads / self.steps
        attributions = (input_tensor - baseline) * avg_grads

        saliency = attributions[0].abs().amax(dim=0).cpu().numpy()
        saliency -= saliency.min()
        max_val = saliency.max()
        if max_val > 1e-8:
            saliency /= max_val
        return saliency
