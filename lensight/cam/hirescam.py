"""
HiResCAM: High-Resolution Class Activation Mapping.

Draelos & Carin, 2020 (NeurIPS, https://arxiv.org/abs/2011.08891).

Standard Grad-CAM averages gradients across the entire spatial dimension
before weighting feature maps. This spatial averaging can cause Grad-CAM to
highlight regions that do not actually contribute to the model's prediction.
HiResCAM retains elementwise gradients:
    L_HiResCAM = ReLU( sum_k ( dY^c / dA^k ) * A^k )

Guarantees that each spatial location reflects only the gradient at that
specific location, providing higher fidelity for fine-grained localization.
"""

from __future__ import annotations
import torch

from ..core.base import BaseCAM


class HiResCAM(BaseCAM):
    """
    High-Resolution CAM using elementwise gradient multiplication without
    spatial average pooling.

    Example
    -------
        cam = HiResCAM(model)
        heatmap = cam.explain(image, target_class=3)
    """

    def _compute_weights(self, activations, gradients, scores, target_class):
        # Return gradients directly without spatial pooling -> (C, h, w)
        if gradients is None:
            return torch.ones_like(activations)
        return gradients
