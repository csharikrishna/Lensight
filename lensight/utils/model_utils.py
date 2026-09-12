"""Helpers for introspecting an arbitrary PyTorch CV model."""

from __future__ import annotations
import torch
import torch.nn as nn


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Return the last ``nn.Conv2d`` module found while walking the model in
    execution order. This is the layer CAM-based methods target by default,
    since it is the last point at which spatial (H x W) information survives
    before pooling/flattening collapses it.

    Raises
    ------
    ValueError
        If the model contains no Conv2d layers (e.g. a pure MLP or a
        transformer with no convolutional stem), in which case the caller
        must pass ``target_layer`` explicitly.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError(
            "No nn.Conv2d layer found in this model. CAM-based methods need "
            "a convolutional feature map to explain — pass `target_layer=` "
            "explicitly (e.g. the last block of a custom backbone)."
        )
    return last_conv


def find_last_linear_layer(model: nn.Module) -> nn.Module:
    """
    Return the last ``nn.Linear`` module in execution order — typically the
    final classifier head. Used to tap the penultimate-layer embedding (the
    input to this layer) for clustering-based failure analysis.
    """
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError(
            "No nn.Linear layer found in this model — cannot auto-locate a "
            "classifier head for embedding extraction. Pass `embedding_layer=` "
            "explicitly."
        )
    return last_linear


def get_device(model: nn.Module) -> torch.device:
    """
    Return the device the model's parameters currently live on. Falls back to
    CPU for parameter-free models (e.g. a purely rule-based or lookup-table
    "model" used in tests) rather than raising StopIteration.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def named_module(model: nn.Module, dotted_name: str) -> nn.Module:
    """Fetch a submodule by dotted path, e.g. 'layer4.1.conv2'."""
    module = model
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def create_vit_reshape_transform(has_cls_token: bool = True):
    """
    Factory creating a reshape transform for Vision Transformers (ViT, DeiT, DINO, etc.).
    Converts token sequence ``(B, N, C)`` into spatial 2D feature maps ``(B, C, H, W)``.

    Parameters
    ----------
    has_cls_token : bool, default=True
        Whether the first token in the sequence is the class ([CLS]) token.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor] compatible with ``BaseCAM(..., reshape_transform=...)``.
    """
    def _reshape(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 4:
            # Already 4D (B, C, H, W)
            return tensor
        # Assume tensor is (B, N, C)
        if has_cls_token and tensor.shape[1] > 1:
            patches = tensor[:, 1:, :]
        else:
            patches = tensor
        b, n, c = patches.shape
        h = int(n ** 0.5)
        w = int(n / h)
        if h * w != n:
            h = int(round(n ** 0.5))
            w = h
            patches = patches[:, : h * w, :]
        return patches.transpose(1, 2).reshape(b, c, h, w)

    return _reshape


def find_target_layer(model: nn.Module) -> nn.Module:
    """
    Find the best candidate layer for CAM explanation.
    Checks for nn.Conv2d first. If none is found, looks for LayerNorm or
    feature backbones.
    """
    last_conv = None
    last_norm = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            last_norm = module
    if last_conv is not None:
        return last_conv
    if last_norm is not None:
        return last_norm

    children = list(model.children())
    if len(children) > 1:
        return children[-2]

    raise ValueError(
        "Could not automatically locate a Conv2d or normalization layer. "
        "Pass `target_layer=` explicitly (e.g. `GradCAM(model, target_layer=model.backbone[-1])`)."
    )
