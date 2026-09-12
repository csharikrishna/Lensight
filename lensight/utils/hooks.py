"""Forward/backward hook management for extracting activations and gradients."""

from __future__ import annotations
import torch
import torch.nn as nn


class ActivationsAndGradients:
    """
    Registers a forward hook and a full-backward hook on a target layer so that,
    after a forward + backward pass, the layer's activations and the gradient
    of the loss with respect to those activations are both available.

    This is the shared plumbing used by every CAM-style explainer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = []
        self._handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] has the same shape as the layer's output.
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.activations = None
        self.gradients = None
        return self.model(x)

    def release(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
