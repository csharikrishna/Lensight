"""Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.

Chattopadhay et al., 2018 (https://arxiv.org/abs/1710.11063). Produces tighter,
better-localized heatmaps than vanilla Grad-CAM, especially useful when an
image contains multiple instances of the target class.
"""

from __future__ import annotations
import torch

from ..core.base import BaseCAM


class GradCAMPlusPlus(BaseCAM):
    """
    Uses a second-order weighting of the gradients (rather than a plain
    global average) so that pixels contributing more non-linearly to the
    class score are emphasized. Best default choice when Grad-CAM heatmaps
    look too diffuse or miss smaller objects.
    """

    def _compute_weights(self, activations, gradients, scores, target_class):
        grads = gradients
        grads_sq = grads.pow(2)
        grads_cube = grads_sq * grads

        sum_acts = activations.sum(dim=(1, 2), keepdim=True)
        eps = 1e-8
        alpha_denom = 2 * grads_sq + sum_acts * grads_cube
        alpha_denom = torch.where(
            alpha_denom != 0, alpha_denom, torch.full_like(alpha_denom, eps)
        )
        alpha = grads_sq / alpha_denom

        positive_grads = torch.relu(grads)
        weights = (alpha * positive_grads).sum(dim=(1, 2))
        return weights
