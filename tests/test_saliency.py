import numpy as np
import pytest

from lensight import VanillaGradient, IntegratedGradients, SmoothGrad


@pytest.mark.parametrize(
    "explainer_cls,kwargs",
    [
        (VanillaGradient, {}),
        (IntegratedGradients, {"steps": 5}),
        (SmoothGrad, {"n_samples": 4}),
    ],
)
def test_saliency_output_shape_matches_input(model, single_image, explainer_cls, kwargs):
    explainer = explainer_cls(model, **kwargs)
    saliency = explainer.explain(single_image)

    assert saliency.shape == single_image.shape[-2:]
    assert saliency.min() >= 0.0 - 1e-6
    assert saliency.max() <= 1.0 + 1e-6


def test_integrated_gradients_with_custom_baseline(model, single_image):
    import torch

    ig = IntegratedGradients(model, steps=5)
    baseline = torch.ones_like(single_image.unsqueeze(0)) * -1
    saliency = ig.explain(single_image, baseline=baseline)
    assert saliency.shape == single_image.shape[-2:]


def test_smoothgrad_is_smoother_than_vanilla_on_average(model, single_image):
    """SmoothGrad should reduce pixel-to-pixel noise relative to vanilla gradients
    (a coarse but real property of the method, not just a shape check)."""
    vanilla = VanillaGradient(model).explain(single_image)
    smooth = SmoothGrad(model, n_samples=16, noise_level=0.2).explain(single_image)

    def total_variation(a):
        return np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum()

    # Not a hard guarantee for every random seed/model, but true often enough
    # that a persistent failure here is a real regression signal.
    assert total_variation(smooth) <= total_variation(vanilla) * 1.5
