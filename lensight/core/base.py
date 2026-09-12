"""Shared base class for CAM-family explainers."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.hooks import ActivationsAndGradients
from ..utils.model_utils import find_last_conv_layer, get_device


from typing import Callable


class BaseCAM(ABC):
    """
    Common scaffolding for Grad-CAM-style explainers: hook registration,
    forward/backward pass, target-class resolution, and heatmap post-processing.
    Subclasses implement `_compute_weights` to define how activation channels
    are weighted into the final map.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or find_last_conv_layer(model)
        self.reshape_transform = reshape_transform
        self.device = get_device(model)

    @abstractmethod
    def _compute_weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor | None,
        scores: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """Return a (C,) weight per activation channel (or compatible spatial weights)."""
        raise NotImplementedError

    def _needs_gradients(self) -> bool:
        return True

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute a 2D class-activation heatmap for a single image.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, C, H, W) or (C, H, W) — a single preprocessed image.
        target_class : int, optional
            Class index to explain. Defaults to the model's predicted class.

        Returns
        -------
        np.ndarray of shape (H, W), values in [0, 1].
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.shape[0] != 1:
            raise ValueError(
                f"explain() expects a single image, got batch size {input_tensor.shape[0]}. "
                "Use explain_batch() for multiple images."
            )
        return self.explain_batch(
            input_tensor,
            target_classes=[target_class] if target_class is not None else None,
        )[0]

    def explain_batch(
        self,
        input_batch: torch.Tensor,
        target_classes: list[int | None] | None = None,
    ) -> list[np.ndarray]:
        """
        Compute 2D class-activation heatmaps for a batch of images using a single
        vectorized forward and backward pass for high throughput.

        Parameters
        ----------
        input_batch : torch.Tensor of shape (B, C, H, W)
        target_classes : list of int or None, optional

        Returns
        -------
        list of np.ndarray, each of shape (H, W) with values in [0, 1].
        """
        if input_batch.dim() == 3:
            input_batch = input_batch.unsqueeze(0)
        b_size = input_batch.shape[0]
        input_batch = input_batch.to(self.device)

        if type(self).explain != BaseCAM.explain:
            n = input_batch.shape[0]
            targets = target_classes or [None] * n
            return [
                self.explain(input_batch[i], target_class=targets[i]) for i in range(n)
            ]

        with ActivationsAndGradients(self.model, self.target_layer) as hook:
            if self._needs_gradients():
                input_batch = input_batch.clone().requires_grad_(True)
                scores = hook(input_batch)
            else:
                with torch.no_grad():
                    scores = hook(input_batch)

            # Resolve target classes
            resolved_targets: list[int] = []
            for i in range(b_size):
                if target_classes is not None and i < len(target_classes) and target_classes[i] is not None:
                    resolved_targets.append(int(target_classes[i]))
                else:
                    resolved_targets.append(int(scores[i].argmax(dim=0).item()))

            if self._needs_gradients():
                self.model.zero_grad(set_to_none=True)
                target_tensor = torch.tensor(resolved_targets, device=self.device, dtype=torch.long)
                # Summing across independent batch elements computes exact gradients
                # for each sample simultaneously
                loss = scores[torch.arange(b_size, device=self.device), target_tensor].sum()
                loss.backward()

            activations = hook.activations
            gradients = hook.gradients

            if self.reshape_transform is not None:
                activations = self.reshape_transform(activations)
                if gradients is not None:
                    gradients = self.reshape_transform(gradients)

        heatmaps: list[np.ndarray] = []
        for i in range(b_size):
            act_i = activations[i]  # (C, h, w)
            grad_i = gradients[i] if gradients is not None else None
            score_i = scores[i : i + 1]
            t_i = resolved_targets[i]

            weights = self._compute_weights(act_i, grad_i, score_i, t_i)
            if weights.dim() == 1:
                cam = torch.relu((weights.view(-1, 1, 1) * act_i).sum(dim=0))
            else:
                # Spatial weights (e.g. HiResCAM elementwise product)
                cam = torch.relu((weights * act_i).sum(dim=0))

            cam_np = cam.detach().cpu().numpy()
            cam_np = cam_np - cam_np.min()
            max_val = cam_np.max()
            if max_val > 1e-8:
                cam_np = cam_np / max_val
            heatmaps.append(cam_np)

        return heatmaps
