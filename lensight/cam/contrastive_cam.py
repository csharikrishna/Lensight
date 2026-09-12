"""
Contrastive Class Activation Mapping.

Answers the question: "Why did the model predict class A instead of class B?"
Standard CAM highlights all features supporting class A, which often includes
shared background or common category attributes. Contrastive CAM backpropagates
the difference:
    L_contrast = Score(class_A) - Score(class_B)

This isolates the discriminative features that uniquely pushed the model
toward the target class over the competing candidate.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ..core.base import BaseCAM
from ..utils.hooks import ActivationsAndGradients


class ContrastiveCAM(BaseCAM):
    """
    Contrastive CAM explaining why target_class was selected over contrast_class.

    If contrast_class is not provided, it automatically selects the runner-up class
    (the class with the second-highest logit).

    Example
    -------
        cam = ContrastiveCAM(model)
        # Explains why class 0 ("cat") was chosen over class 1 ("dog"):
        heatmap = cam.explain(image, target_class=0, contrast_class=1)
    """

    def _compute_weights(self, activations, gradients, scores, target_class):
        if gradients is None:
            return torch.zeros(activations.shape[0], device=activations.device)
        return gradients.mean(dim=(1, 2))

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        contrast_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute a contrastive heatmap for target_class vs contrast_class.

        Parameters
        ----------
        input_tensor : torch.Tensor of shape (1, C, H, W) or (C, H, W)
        target_class : int, optional
            Primary class. Defaults to top predicted class.
        contrast_class : int, optional
            Comparison class. Defaults to the runner-up predicted class.

        Returns
        -------
        np.ndarray of shape (H, W) in [0, 1].
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.shape[0] != 1:
            raise ValueError(
                f"explain() expects a single image, got batch size {input_tensor.shape[0]}."
            )
        input_tensor = input_tensor.to(self.device)

        with ActivationsAndGradients(self.model, self.target_layer) as hook:
            input_tensor = input_tensor.clone().requires_grad_(True)
            scores = hook(input_tensor)

            if target_class is None:
                target_class = int(scores.argmax(dim=1).item())

            if contrast_class is None:
                # Find runner-up class
                scores_clone = scores.clone()
                scores_clone[0, target_class] = float("-inf")
                contrast_class = int(scores_clone.argmax(dim=1).item())

            self.model.zero_grad(set_to_none=True)
            contrast_loss = scores[0, target_class] - scores[0, contrast_class]
            contrast_loss.backward()

            activations = hook.activations
            gradients = hook.gradients

            if self.reshape_transform is not None:
                activations = self.reshape_transform(activations)
                if gradients is not None:
                    gradients = self.reshape_transform(gradients)

            act_0 = activations[0]
            grad_0 = gradients[0] if gradients is not None else None

        weights = self._compute_weights(act_0, grad_0, scores, target_class)
        cam = torch.relu((weights.view(-1, 1, 1) * act_0).sum(dim=0))

        cam_np = cam.detach().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        max_val = cam_np.max()
        if max_val > 1e-8:
            cam_np = cam_np / max_val
        return cam_np
