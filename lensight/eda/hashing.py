"""Perceptual hashing and transitive clustering algorithms for computer vision datasets.

Provides fast, pure-NumPy implementations of:
  - Difference Hash (dhash)
  - Average Hash (ahash)
  - Perceptual DCT Hash (phash)
  - Vectorized pairwise Hamming distance matrices
  - Transitive Union-Find duplicate clustering with smart representative selection
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# Precomputed 256-element lookup table for fast 8-bit popcount
_POPCOUNT_8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _to_pil_grayscale(image_input: Any, size: Tuple[int, int]) -> Image.Image:
    """Normalize input (filepath, PIL Image, numpy array, torch.Tensor) to grayscale PIL."""
    if isinstance(image_input, str):
        with Image.open(image_input) as img:
            return img.convert("L").resize(size, Image.Resampling.BILINEAR)
    elif isinstance(image_input, Image.Image):
        return image_input.convert("L").resize(size, Image.Resampling.BILINEAR)
    elif hasattr(image_input, "cpu"):  # PyTorch Tensor
        tensor = image_input.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        arr = tensor.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(-1)
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            if arr.max() <= 1.05 and arr.min() >= -0.05:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        return img.convert("L").resize(size, Image.Resampling.BILINEAR)
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(-1)
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.05 and arr.min() >= -0.05:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        return img.convert("L").resize(size, Image.Resampling.BILINEAR)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def compute_dhash(image_input: Any, hash_size: int = 8) -> int:
    """Compute difference hash (dhash) as a 64-bit integer.

    Resizes to (hash_size + 1, hash_size), then compares adjacent pixels horizontally.
    Captures relative brightness gradients across columns.
    """
    img = _to_pil_grayscale(image_input, (hash_size + 1, hash_size))
    pixels = np.asarray(img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def compute_ahash(image_input: Any, hash_size: int = 8) -> int:
    """Compute average hash (ahash) as a 64-bit integer.

    Resizes to (hash_size, hash_size), finds mean intensity, and sets bits based on > mean.
    Captures overall low-frequency intensity layout.
    """
    img = _to_pil_grayscale(image_input, (hash_size, hash_size))
    pixels = np.asarray(img, dtype=np.float32)
    mean_val = pixels.mean()
    bits = (pixels > mean_val).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def _create_dct_matrix(n: int) -> np.ndarray:
    """Compute orthonormal 1D Discrete Cosine Transform (DCT-II) matrix of size n."""
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == 0:
                matrix[i, j] = 1.0 / math.sqrt(n)
            else:
                matrix[i, j] = math.sqrt(2.0 / n) * math.cos((2 * j + 1) * i * math.pi / (2.0 * n))
    return matrix


_DCT_32 = _create_dct_matrix(32)


def compute_phash(image_input: Any, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Compute perceptual DCT hash (phash) as a 64-bit integer.

    Resizes to (32, 32), applies 2D DCT, takes top-left 8x8 frequency block,
    and sets bits based on whether each coefficient is above the block median.
    Robust against minor edits, scaling, compression, and contrast shifts.
    """
    img_size = hash_size * highfreq_factor  # 32
    img = _to_pil_grayscale(image_input, (img_size, img_size))
    pixels = np.asarray(img, dtype=np.float32)

    # 2D DCT via matrix multiplication: D * pixels * D.T
    dct = _DCT_32 @ pixels @ _DCT_32.T
    sub_block = dct[:hash_size, :hash_size]

    # Exclude DC term (0, 0) when computing median for contrast invariance
    med = np.median(sub_block.flatten()[1:])
    bits = (sub_block > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


HASH_FUNCTIONS = {
    "dhash": compute_dhash,
    "ahash": compute_ahash,
    "phash": compute_phash,
}


def compute_hash(image_input: Any, algo: str = "dhash") -> int:
    """Compute perceptual hash using chosen algorithm ('dhash', 'ahash', or 'phash')."""
    if algo not in HASH_FUNCTIONS:
        raise ValueError(f"Unknown hash algorithm '{algo}'. Choose from {list(HASH_FUNCTIONS)}")
    return HASH_FUNCTIONS[algo](image_input)


def hamming_distance(a: int, b: int) -> int:
    """Compute bitwise Hamming distance between two 64-bit hashes."""
    return (a ^ b).bit_count()


def pairwise_hamming_matrix(hashes_a: Sequence[int], hashes_b: Sequence[int]) -> np.ndarray:
    """Compute pairwise Hamming distance matrix between two lists of 64-bit hashes.

    Returns uint8 array of shape (len(hashes_a), len(hashes_b)).
    Uses vectorized NumPy bit-manipulation and popcount lookup table for speed.
    """
    a_arr = np.array(hashes_a, dtype=np.uint64)
    b_arr = np.array(hashes_b, dtype=np.uint64)

    # XOR broadcast: shape (N, M)
    diff = a_arr[:, None] ^ b_arr[None, :]

    # View as 8 uint8 bytes: shape (N, M, 8)
    byte_view = diff.view(np.uint8).reshape(len(a_arr), len(b_arr), 8)

    # Lookup 8-bit popcount and sum across bytes
    dist_matrix = _POPCOUNT_8[byte_view].sum(axis=-1).astype(np.uint8)
    return dist_matrix


class UnionFind:
    """Disjoint-set data structure with path compression and union by rank."""

    def __init__(self):
        self.parent: dict[Any, Any] = {}
        self.rank: dict[Any, int] = {}

    def find(self, item: Any) -> Any:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        curr = item
        while self.parent[curr] != root:
            next_node = self.parent[curr]
            self.parent[curr] = root
            curr = next_node
        return root

    def union(self, a: Any, b: Any) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1


@dataclass
class DuplicateGroup:
    """A cluster of transitively duplicate samples.

    Attributes:
        cluster_id: Unique string identifier (e.g. 'c1', 'Trouser-1').
        survivor_id: ID of the kept representative sample.
        duplicate_ids: IDs of redundant duplicate samples in this cluster.
        distances_to_survivor: Map from duplicate_id to Hamming distance to survivor.
        category: Class name or group tag if known.
    """
    cluster_id: str
    survivor_id: Any
    duplicate_ids: List[Any]
    distances_to_survivor: dict[Any, int]
    category: Optional[str] = None

    @property
    def total_count(self) -> int:
        return 1 + len(self.duplicate_ids)


def cluster_transitive_duplicates(
    items: Sequence[Any],
    hashes: Sequence[int],
    threshold: int = 3,
    quality_scores: Optional[Sequence[float]] = None,
    category: Optional[str] = None,
) -> List[DuplicateGroup]:
    """Cluster duplicate samples transitively using Union-Find.

    Unlike naive approaches that arbitrarily pick the first sample, this function
    supports smart survivor selection using quality_scores (e.g., Laplacian sharpness,
    resolution, or contrast). The highest-quality member is chosen as the survivor.

    Args:
        items: List of item identifiers (file paths, integer indices, or keys).
        hashes: Corresponding 64-bit perceptual hashes.
        threshold: Maximum Hamming distance to consider two images duplicates (default: 3).
        quality_scores: Optional quality scores (higher is better). If omitted, picks first item.
        category: Optional class label or category tag.

    Returns:
        List of DuplicateGroup objects for clusters with 2+ members.
    """
    n = len(items)
    if n < 2:
        return []

    # Compute pairwise Hamming distances in fast vectorized batches
    uf = UnionFind()
    hash_arr = np.array(hashes, dtype=np.uint64)

    # Chunked pairwise comparison to conserve memory on large datasets
    chunk_size = 500
    for start_i in range(0, n, chunk_size):
        end_i = min(start_i + chunk_size, n)
        chunk_a = hash_arr[start_i:end_i]

        for start_j in range(start_i, n, chunk_size):
            end_j = min(start_j + chunk_size, n)
            chunk_b = hash_arr[start_j:end_j]

            diff = chunk_a[:, None] ^ chunk_b[None, :]
            byte_view = diff.view(np.uint8).reshape(len(chunk_a), len(chunk_b), 8)
            dists = _POPCOUNT_8[byte_view].sum(axis=-1)

            # Find matching pairs (i < j)
            mask = dists <= threshold
            if start_i == start_j:
                diag_mask = np.triu(np.ones_like(mask, dtype=bool), k=1)
                mask &= diag_mask

            matches = np.argwhere(mask)
            for local_i, local_j in matches:
                global_i = start_i + local_i
                global_j = start_j + local_j
                uf.union(items[global_i], items[global_j])

    # Group items by connected component root
    item_to_hash = {items[i]: hashes[i] for i in range(n)}
    item_to_quality = (
        {items[i]: quality_scores[i] for i in range(n)}
        if quality_scores is not None
        else {items[i]: 0.0 for i in range(n)}
    )

    clusters_map: dict[Any, List[Any]] = defaultdict(list)
    for it in items:
        clusters_map[uf.find(it)].append(it)

    result_clusters: List[DuplicateGroup] = []
    cluster_num = 1
    for root, members in clusters_map.items():
        if len(members) < 2:
            continue

        # Smart survivor selection: choose member with highest quality score
        # Break ties deterministically by string representation of item
        members.sort(key=lambda m: (-item_to_quality[m], str(m)))
        survivor = members[0]
        duplicates = members[1:]

        survivor_hash = item_to_hash[survivor]
        distances = {d: hamming_distance(survivor_hash, item_to_hash[d]) for d in duplicates}

        cid = f"{category}-{cluster_num}" if category else f"cluster-{cluster_num}"
        result_clusters.append(
            DuplicateGroup(
                cluster_id=cid,
                survivor_id=survivor,
                duplicate_ids=duplicates,
                distances_to_survivor=distances,
                category=category,
            )
        )
        cluster_num += 1

    return result_clusters
