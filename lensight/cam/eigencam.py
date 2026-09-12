"""Eigen-CAM: Class Activation Map using Principal Components.

Muhammad & Yeasin, 2020 (https://arxiv.org/abs/2008.00299). Gradient-free —
it explains "what the layer is looking at" via the dominant principal
component of the activations, without needing a backward pass or even a
target class. Useful as a sanity check that doesn't depend on gradient
quality (e.g. with quantized or ReLU6-clipped models where gradients can be
uninformative).
"""

from __future__ import annotations
import torch

from ..core.base import BaseCAM


class EigenCAM(BaseCAM):
    """Projects activations onto their first principal component (via SVD)."""

    def _needs_gradients(self) -> bool:
        return False

    def _compute_weights(self, activations, gradients, scores, target_class):
        # Not used directly — EigenCAM overrides the map computation below
        # rather than expressing itself as a per-channel weight vector, since
        # the projection mixes channels rather than summing them.
        raise NotImplementedError

    def explain(self, input_tensor, target_class=None):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        from ..utils.hooks import ActivationsAndGradients

        with ActivationsAndGradients(self.model, self.target_layer) as hook, torch.no_grad():
            hook(input_tensor)
            activations = hook.activations
            if self.reshape_transform is not None:
                activations = self.reshape_transform(activations)
            activations = activations[0]  # (C, h, w)

        c, h, w = activations.shape
        flat = activations.reshape(c, h * w).cpu()
        flat = flat - flat.mean(dim=1, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(flat, full_matrices=False)
            principal = vh[0]  # (h*w,)
        except RuntimeError:
            # SVD can fail to converge on degenerate (e.g. all-zero) activations.
            principal = flat.mean(dim=0)

        cam = principal.reshape(h, w)
        cam = torch.relu(cam)
        cam_np = cam.numpy()
        cam_np = cam_np - cam_np.min()
        max_val = cam_np.max()
        if max_val > 1e-8:
            cam_np = cam_np / max_val
        return cam_np
