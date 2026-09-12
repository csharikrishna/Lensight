import json
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from lensight import (
    HiResCAM,
    ContrastiveCAM,
    GradCAM,
    TemperatureScaler,
    CalibratedModel,
    create_vit_reshape_transform,
    MisclassificationAnalyzer,
    ModelDoctor,
)


def test_temperature_scaler(model, dataloader):
    scaler = TemperatureScaler(model)
    summary = scaler.fit(dataloader, max_iter=20)

    assert summary.temperature > 0
    assert 0 <= summary.calibrated_ece <= 1.0
    assert 0 <= summary.calibrated_mce <= 1.0
    summary_dict = summary.to_dict()
    assert "temperature" in summary_dict
    assert "ece_improvement_pct" in summary_dict
    assert "Temperature:" in summary.summary_text()

    cal_model = scaler.calibrated_model
    assert isinstance(cal_model, CalibratedModel)

    # Verify prediction invariance (top-1 class is identical)
    test_img = torch.randn(2, 3, 16, 16)
    orig_pred = model(test_img).argmax(dim=-1)
    cal_pred = cal_model(test_img).argmax(dim=-1)
    assert torch.equal(orig_pred, cal_pred)

    probs = cal_model.predict_proba(test_img)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_hirescam(model, single_image):
    hires = HiResCAM(model)
    cam = hires.explain(single_image)
    assert isinstance(cam, np.ndarray)
    assert cam.ndim == 2
    assert 0.0 <= cam.min() <= cam.max() <= 1.0


def test_contrastive_cam(model, single_image):
    ccam = ContrastiveCAM(model)

    # Auto runner-up
    cam_auto = ccam.explain(single_image, target_class=0)
    assert isinstance(cam_auto, np.ndarray)
    assert cam_auto.ndim == 2
    assert 0.0 <= cam_auto.min() <= cam_auto.max() <= 1.0

    # Explicit contrast class
    cam_explicit = ccam.explain(single_image, target_class=0, contrast_class=1)
    assert isinstance(cam_explicit, np.ndarray)
    assert cam_explicit.ndim == 2
    assert 0.0 <= cam_explicit.min() <= cam_explicit.max() <= 1.0


def test_batched_cam_consistency(model):
    batch = torch.randn(3, 3, 16, 16)
    cam = GradCAM(model)

    # Vectorized batch
    batch_maps = cam.explain_batch(batch, target_classes=[0, 1, 2])
    assert len(batch_maps) == 3

    # Single execution
    single_maps = [cam.explain(batch[i], target_class=i) for i in range(3)]

    for b_map, s_map in zip(batch_maps, single_maps):
        np.testing.assert_allclose(b_map, s_map, atol=1e-4)


def test_label_error_detection(model):
    # Construct a dataset where ground truth is wrong for high confidence inputs
    images = torch.randn(10, 3, 16, 16)
    # Give all inputs label 0, but model will have varying predictions
    labels = torch.zeros(10, dtype=torch.long)
    ds = data.TensorDataset(images, labels)
    loader = data.DataLoader(ds, batch_size=5)

    analyzer = MisclassificationAnalyzer(model, num_classes=4)
    report = analyzer.analyze(loader, label_error_threshold=0.01)

    errors = report.find_label_errors(min_confidence=0.01)
    assert isinstance(errors, list)
    report_dict = report.to_dict()
    assert "suspected_label_errors" in report_dict


def test_vit_reshape_transform():
    # Simulate ViT tokens: (Batch=2, Tokens=1 + 16, Dim=32)
    transform = create_vit_reshape_transform(has_cls_token=True)
    tokens = torch.randn(2, 17, 32)
    spatial = transform(tokens)
    assert spatial.shape == (2, 32, 4, 4)

    # Without CLS token: (Batch=2, Tokens=16, Dim=32)
    transform_no_cls = create_vit_reshape_transform(has_cls_token=False)
    tokens_no_cls = torch.randn(2, 16, 32)
    spatial_no_cls = transform_no_cls(tokens_no_cls)
    assert spatial_no_cls.shape == (2, 32, 4, 4)


def test_doctor_serialization_and_notebook(model, dataloader, tmp_path):
    doctor = ModelDoctor(model, class_names=["a", "b", "c", "d"])
    report = doctor.diagnose(dataloader, n_clusters=2, n_example_images=4)

    # HTML representation in notebook
    repr_html = report._repr_html_()
    assert "<iframe" in repr_html
    assert "srcdoc=" in repr_html

    # to_dict & to_json
    data_dict = report.to_dict()
    assert "misclassification" in data_dict
    assert "calibration" in data_dict

    json_path = tmp_path / "diagnosis.json"
    report.to_json(str(json_path))
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["misclassification"]["accuracy"] == report.misclassification.accuracy

    # HTML contains interactive filter chips and dashboard elements
    assert "filter-chips" in report.html
    assert "gallery-card" in report.html


def test_lensight_help_and_cli(capsys):
    import lensight
    lensight.help()
    captured = capsys.readouterr()
    assert "LENSIGHT QUICK REFERENCE" in captured.out
    assert "Visual Explanations" in captured.out

    from lensight.cli import main
    main(["info"])
    captured_info = capsys.readouterr()
    assert "Lensight Environment Status" in captured_info.out

    main(["help"])
    captured_help = capsys.readouterr()
    assert "LENSIGHT QUICK REFERENCE" in captured_help.out

