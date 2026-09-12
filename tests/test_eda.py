import numpy as np
import pytest
import torch
import torch.utils.data as data

from lensight import DatasetExplorer, EDAReport, ModelDoctor


def test_dataset_explorer_basic():
    # 20 samples, 2 classes with intentional 3:1 imbalance
    imgs = torch.randn(20, 3, 16, 16)
    lbls = torch.tensor([0] * 15 + [1] * 5)
    ds = data.TensorDataset(imgs, lbls)

    explorer = DatasetExplorer(ds, class_names=["dog", "cat"])
    report = explorer.analyze()

    assert isinstance(report, EDAReport)
    assert report.total_images == 20
    assert report.class_counts == {0: 15, 1: 5}
    assert report.imbalance_ratio == 3.0
    assert report.is_imbalanced is True
    assert "dog" in report.summary_text()
    assert "cat" in report.summary_text()

    d = report.to_dict()
    assert "class_distribution" in d
    assert "exposure" in d
    assert "sharpness" in d
    assert "integrity" in d


def test_exposure_and_outlier_detection():
    # Create distinct images: 1 normal, 1 dark, 1 bright, 1 low contrast
    imgs = torch.zeros(4, 3, 16, 16)
    imgs[0] = 0.5  # normal
    imgs[1] = 0.02  # underexposed (dark)
    imgs[2] = 0.98  # overexposed (bright)
    imgs[3] = torch.randn(3, 16, 16) * 0.01 + 0.5  # low contrast

    lbls = torch.tensor([0, 0, 1, 1])
    ds = data.TensorDataset(imgs, lbls)

    explorer = DatasetExplorer(ds)
    report = explorer.analyze()

    assert 1 in report.underexposed_indices
    assert 2 in report.overexposed_indices
    assert 3 in report.low_contrast_indices


def test_duplicate_detection():
    # Create 4 images where image 0 and image 2 are identical
    imgs = torch.randn(4, 3, 16, 16)
    imgs[2] = imgs[0].clone()
    lbls = torch.tensor([0, 1, 0, 1])
    ds = data.TensorDataset(imgs, lbls)

    explorer = DatasetExplorer(ds)
    report = explorer.analyze()

    assert len(report.duplicate_pairs) >= 1
    pair = report.duplicate_pairs[0]
    assert (pair == (0, 2)) or (pair == (2, 0))


def test_html_reports(tmp_path):
    imgs = torch.randn(8, 3, 16, 16)
    lbls = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    ds = data.TensorDataset(imgs, lbls)

    explorer = DatasetExplorer(ds, class_names=["a", "b", "c"])
    report = explorer.analyze()

    # Notebook representation
    repr_html = report._repr_html_()
    assert "<iframe" in repr_html

    # Save HTML
    out_file = tmp_path / "eda_test.html"
    report.save_html(str(out_file))
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "LENSIGHT DATASET PROFILER" in content
    assert "Class Distribution" in content


def test_model_doctor_eda_integration(model, dataloader):
    doctor = ModelDoctor(model, class_names=["c0", "c1", "c2", "c3"])
    report = doctor.diagnose(dataloader, include_eda=True, max_eda_samples=30)

    assert report.eda is not None
    assert isinstance(report.eda, EDAReport)
    assert report.eda.total_images == 30
    assert "Dataset Exploratory Health (EDA)" in report.html
    assert "eda" in report.to_dict()
