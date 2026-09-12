import numpy as np
import pytest

from lensight import GradCAM, GradCAMPlusPlus, EigenCAM


@pytest.mark.parametrize("cam_cls", [GradCAM, GradCAMPlusPlus, EigenCAM])
def test_cam_output_shape_and_range(model, single_image, cam_cls):
    cam = cam_cls(model)
    heatmap = cam.explain(single_image)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0 - 1e-6
    assert heatmap.max() <= 1.0 + 1e-6


def test_gradcam_explicit_target_class(model, single_image, num_classes):
    cam = GradCAM(model)
    heatmap_a = cam.explain(single_image, target_class=0)
    heatmap_b = cam.explain(single_image, target_class=num_classes - 1)
    # Different target classes should not always yield an identical map.
    assert not np.allclose(heatmap_a, heatmap_b)


def test_gradcam_batch_of_one_via_unsqueezed_input(model, single_image):
    cam = GradCAM(model)
    heatmap_3d = cam.explain(single_image)
    heatmap_4d = cam.explain(single_image.unsqueeze(0))
    assert heatmap_3d.shape == heatmap_4d.shape


def test_explain_batch_rejects_multi_image_explain(model):
    import torch

    cam = GradCAM(model)
    batch = torch.randn(3, 3, 16, 16)
    with pytest.raises(ValueError):
        cam.explain(batch)


def test_explain_batch_returns_one_heatmap_per_image(model):
    import torch

    cam = GradCAM(model)
    batch = torch.randn(3, 3, 16, 16)
    heatmaps = cam.explain_batch(batch)
    assert len(heatmaps) == 3
    for hm in heatmaps:
        assert hm.ndim == 2


def test_explicit_target_layer(model, single_image):
    target_layer = model.features[2]  # second conv layer
    cam = GradCAM(model, target_layer=target_layer)
    heatmap = cam.explain(single_image)
    assert heatmap.ndim == 2


def test_no_conv_layer_raises():
    import torch.nn as nn

    mlp = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 4))
    with pytest.raises(ValueError):
        GradCAM(mlp)
