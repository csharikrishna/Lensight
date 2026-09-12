"""Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

Selvaraju et al., 2017 (https://arxiv.org/abs/1610.02391).
"""

from __future__ import annotations
import torch

from ..core.base import BaseCAM


class GradCAM(BaseCAM):
    """
    Weights each activation channel by the global-average-pooled gradient of
    the target class score with respect to that channel. Works with any CNN
    that exposes at least one Conv2d layer — no architecture changes needed.

    Example
    -------
        cam = GradCAM(model)                     # auto-picks last conv layer
        heatmap = cam.explain(image, target_class=5)
    """

    def _compute_weights(self, activations, gradients, scores, target_class):
        # Global average pool the gradients over spatial dims -> per-channel weight.
        return gradients.mean(dim=(1, 2))
