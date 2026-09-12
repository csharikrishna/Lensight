"""Unit tests for DatasetAuditor, Perceptual Hashing, Leakage, Cross-label detection, and Sanitization."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
from PIL import Image

from lensight.eda.hashing import (
    compute_dhash,
    compute_ahash,
    compute_phash,
    hamming_distance,
    pairwise_hamming_matrix,
    cluster_transitive_duplicates,
)
from lensight.eda.auditor import (
    DatasetAuditor,
    audit_dataset,
    check_leakage,
    check_cross_label,
)
from lensight.eda.sanitizer import DatasetSanitizer


def _create_synthetic_image(pattern="box", color=(128, 128, 128), size=(64, 64)):
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if pattern == "box":
        arr[10:30, 10:30, :] = 255
    elif pattern == "stripe":
        arr[:, 20:40, :] = 255
    elif pattern == "diag":
        for i in range(min(size)):
            arr[i, i, :] = 255
    return Image.fromarray(arr)


def test_hashing_and_hamming_matrix():
    im1 = _create_synthetic_image(pattern="box", color=(200, 50, 50))
    im2 = _create_synthetic_image(pattern="box", color=(200, 50, 50))  # identical
    im3 = _create_synthetic_image(pattern="stripe", color=(10, 200, 10))   # different

    h1 = compute_dhash(im1)
    h2 = compute_dhash(im2)
    h3 = compute_dhash(im3)

    assert hamming_distance(h1, h2) == 0, "Identical images must have 0 Hamming distance"
    assert hamming_distance(h1, h3) > 0, "Different patterns must produce distinct hashes"

    # Vectorized matrix
    hashes = [h1, h2, h3]
    mat = pairwise_hamming_matrix(hashes, hashes)
    assert mat.shape == (3, 3)
    assert mat[0, 1] == 0
    assert mat[1, 0] == 0
    assert mat[0, 2] == hamming_distance(h1, h3)


def test_transitive_clustering_and_smart_survivor():
    # Item A ~ B (dist 1), B ~ C (dist 1), but dist(A, C) = 2
    # Transitive Union-Find should cluster {A, B, C} into a single group!
    items = ["img_a", "img_b", "img_c"]
    # Hand-craft 3 hashes where A-B is 1 bit, B-C is 1 bit
    h_a = 0b0000
    h_b = 0b0001
    h_c = 0b0011
    hashes = [h_a, h_b, h_c]
    qualities = [10.0, 50.0, 20.0]  # B has highest quality

    clusters = cluster_transitive_duplicates(
        items=items,
        hashes=hashes,
        threshold=1,
        quality_scores=qualities,
        category="TestClass",
    )

    assert len(clusters) == 1
    cl = clusters[0]
    assert cl.survivor_id == "img_b", "Highest quality image must be chosen as the kept survivor"
    assert set(cl.duplicate_ids) == {"img_a", "img_c"}
    assert cl.total_count == 3


def test_audit_dataloader_and_clean_indices():
    # Create mock PyTorch dataset: 20 images of class 0, 5 images of class 1 (imbalance 4.0x)
    # Include 2 identical duplicate pairs in class 0
    dup_tensor = torch.full((3, 32, 32), 0.5, dtype=torch.float32)

    images = []
    labels = []
    # Class 0: 20 samples
    for i in range(20):
        if i in (0, 1):  # duplicate pair
            images.append(dup_tensor)
        else:
            images.append(torch.rand(3, 32, 32))
        labels.append(0)

    # Class 1: 5 samples
    for i in range(5):
        images.append(torch.rand(3, 32, 32))
        labels.append(1)

    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, idx):
            return images[idx], labels[idx]

    dataset = MockDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False)

    auditor = DatasetAuditor(class_names=["cat", "dog"])
    report = auditor.audit(loader, threshold=0)

    assert report.total_samples == 25
    assert report.imbalance_ratio == 4.0
    assert report.class_counts["cat"] == 20
    assert report.class_counts["dog"] == 5
    assert len(report.duplicate_clusters) >= 1

    # In-memory clean indices for PyTorch
    clean_indices = report.get_clean_indices(remove_duplicates=True)
    assert len(clean_indices) < 25
    # One of index 0 or 1 was dropped
    assert not (0 in clean_indices and 1 in clean_indices)

    # Test DatasetSanitizer.clean_subset
    subset = DatasetSanitizer.clean_subset(dataset, report)
    assert len(subset) == len(clean_indices)


def test_train_test_leakage_detection():
    # Leaked sample: identical image in train and test
    leaked_im = torch.full((3, 28, 28), 0.8)
    unique_train = [torch.rand(3, 28, 28) for _ in range(5)]
    unique_test = [torch.rand(3, 28, 28) for _ in range(3)]

    train_data = [(unique_train[i], 0) for i in range(5)] + [(leaked_im, 0)]
    test_data = [(unique_test[i], 0) for i in range(3)] + [(leaked_im, 0)]

    report = check_leakage(train_data, test_data, class_names=["digit"], threshold=0)

    assert report.train_count == 6
    assert report.test_count == 4
    assert report.leaked_count == 1
    assert report.leakage_rate == 0.25
    assert len(report.matches) == 1
    assert report.matches[0].is_exact is True

    # Test clean indices
    clean_test_indices = report.get_clean_test_indices()
    assert 3 not in clean_test_indices
    assert len(clean_test_indices) == 3


def test_cross_label_conflict_detection():
    # Same image assigned to class 0 AND class 1
    conflict_im = torch.full((3, 28, 28), 0.3)

    dataset = [
        (conflict_im, 0),  # cat
        (conflict_im, 1),  # dog (contradiction!)
        (torch.rand(3, 28, 28), 0),
        (torch.rand(3, 28, 28), 1),
    ]

    report = check_cross_label(dataset, class_names=["cat", "dog"], threshold=0)
    assert report.conflict_count == 1
    assert report.conflicts[0].class_a == "cat"
    assert report.conflicts[0].class_b == "dog"
    assert report.conflicts[0].is_exact is True


def test_directory_auditing_and_sanitizer(tmp_path):
    # Build directory structure: dataset/cats and dataset/dogs
    dataset_dir = tmp_path / "dataset"
    cats_dir = dataset_dir / "cats"
    dogs_dir = dataset_dir / "dogs"
    cats_dir.mkdir(parents=True)
    dogs_dir.mkdir(parents=True)

    # Put 2 identical cat images
    cat_im = _create_synthetic_image(pattern="box", color=(100, 100, 100))
    cat1 = str(cats_dir / "cat1.jpg")
    cat2 = str(cats_dir / "cat2.jpg")
    cat_im.save(cat1)
    cat_im.save(cat2)

    # Put 1 unique dog image
    dog_im = _create_synthetic_image(pattern="stripe", color=(20, 20, 200))
    dog1 = str(dogs_dir / "dog1.jpg")
    dog_im.save(dog1)

    # Put 1 corrupt file
    corrupt_file = cats_dir / "broken.jpg"
    corrupt_file.write_bytes(b"not an image file content")

    # Run audit
    report = audit_dataset(str(dataset_dir), threshold=0)

    assert report.total_samples == 3
    assert len(report.corrupt_files) == 1
    assert len(report.duplicate_clusters) == 1

    # Test HTML generation
    html_file = tmp_path / "audit_report.html"
    report.save_html(str(html_file))
    assert html_file.exists()
    assert html_file.stat().st_size > 500

    # Test export clean dataset
    clean_dir = tmp_path / "clean_dataset"
    copied = DatasetSanitizer.export_clean_directory(str(dataset_dir), str(clean_dir), report, mode="copy")
    assert copied == 2  # 1 cat survivor + 1 dog
    assert (clean_dir / "cats").exists()
    assert (clean_dir / "dogs").exists()
    assert len(list((clean_dir / "cats").glob("*.jpg"))) == 1
