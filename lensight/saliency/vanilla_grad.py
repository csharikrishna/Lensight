"""Vanilla gradient saliency maps (Simonyan et al., 2013)."""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ..utils.model_utils import get_device


class VanillaGradient:
    """
    The simplest possible pixel-attribution method: the gradient of the
    target class score with respect to the input pixels. Fast, but noisy —
    prefer IntegratedGradients or SmoothGrad for anything you'll publish.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.device = get_device(model)

    def explain(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device).clone().requires_grad_(True)

        scores = self.model(input_tensor)
        if target_class is None:
            target_class = int(scores.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        scores[0, target_class].backward()

        grad = input_tensor.grad[0].detach().cpu()  # (C, H, W)
        saliency = grad.abs().amax(dim=0).numpy()  # collapse channels -> (H, W)

        saliency -= saliency.min()
        max_val = saliency.max()
        if max_val > 1e-8:
            saliency /= max_val
        return saliency
