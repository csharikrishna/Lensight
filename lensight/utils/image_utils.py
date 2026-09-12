"""Image conversion and heatmap-overlay helpers shared by explainers and reports."""

from __future__ import annotations
import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize(
    tensor: torch.Tensor,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> np.ndarray:
    """
    Convert a normalized CHW tensor (as produced by torchvision transforms)
    back into an HWC uint8 numpy image for display.
    """
    img = tensor.detach().cpu().float().clone()
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    img = img * std_t + mean_t
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return (img * 255).astype(np.uint8)


def to_uint8_image(tensor_or_array) -> np.ndarray:
    """Best-effort conversion of a CHW/HWC tensor or array in [0,1] or [0,255] to uint8 HWC."""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor_or_array, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Map a single-channel float heatmap in [0, 1] to an RGB uint8 image using a
    dependency-free 'jet-like' colormap (avoids requiring OpenCV/matplotlib
    just to colorize an array).
    """
    heatmap = np.clip(heatmap, 0, 1)
    r = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend a colorized heatmap on top of an RGB uint8 image.

    Parameters
    ----------
    image : (H, W, 3) or (H, W, 1) or (H, W) uint8
    heatmap : (h, w) float array in [0, 1], any spatial size (will be resized
        via nearest-neighbour-free bilinear-ish numpy resampling to match image).
    alpha : blend strength for the heatmap.
    """
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    h_img, w_img = image.shape[:2]
    heatmap_resized = _resize_2d(heatmap, h_img, w_img)
    colored = apply_colormap(heatmap_resized).astype(np.float32)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def _resize_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Lightweight bilinear resize for a 2D float array, no cv2/PIL dependency."""
    in_h, in_w = arr.shape
    if (in_h, in_w) == (out_h, out_w):
        return arr
    row_idx = np.linspace(0, in_h - 1, out_h)
    col_idx = np.linspace(0, in_w - 1, out_w)

    r0 = np.floor(row_idx).astype(int)
    r1 = np.clip(r0 + 1, 0, in_h - 1)
    c0 = np.floor(col_idx).astype(int)
    c1 = np.clip(c0 + 1, 0, in_w - 1)

    rw = (row_idx - r0).reshape(-1, 1)
    cw = (col_idx - c0).reshape(1, -1)

    top = arr[r0][:, c0] * (1 - cw) + arr[r0][:, c1] * cw
    bottom = arr[r1][:, c0] * (1 - cw) + arr[r1][:, c1] * cw
    return top * (1 - rw) + bottom * rw
