"""SmoothGrad (Smilkov et al., 2017, https://arxiv.org/abs/1706.03825).

Averages vanilla-gradient saliency maps over several noisy copies of the
input to cancel out the sharp, visually noisy artifacts that raw gradients
tend to produce.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ..utils.model_utils import get_device


class SmoothGrad:
    def __init__(self, model: nn.Module, n_samples: int = 25, noise_level: float = 0.15):
        """
        Parameters
        ----------
        n_samples : number of noisy samples to average over.
        noise_level : std of the Gaussian noise, as a fraction of
            (max - min) of the input tensor's values.
        """
        self.model = model
        self.model.eval()
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.device = get_device(model)

    def explain(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        if target_class is None:
            with torch.no_grad():
                target_class = int(self.model(input_tensor).argmax(dim=1).item())

        value_range = (input_tensor.max() - input_tensor.min()).item()
        sigma = self.noise_level * value_range if value_range > 0 else self.noise_level

        accumulated = torch.zeros_like(input_tensor)
        for _ in range(self.n_samples):
            noise = torch.randn_like(input_tensor) * sigma
            noisy_input = (input_tensor + noise).clone().requires_grad_(True)

            scores = self.model(noisy_input)
            self.model.zero_grad(set_to_none=True)
            scores[0, target_class].backward()

            accumulated += noisy_input.grad.detach()

        avg_grad = accumulated / self.n_samples
        saliency = avg_grad[0].abs().amax(dim=0).cpu().numpy()
        saliency -= saliency.min()
        max_val = saliency.max()
        if max_val > 1e-8:
            saliency /= max_val
        return saliency
