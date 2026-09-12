import os
import tempfile

from lensight import ModelDoctor


def test_diagnose_end_to_end(model, dataloader, num_classes):
    class_names = [f"class_{i}" for i in range(num_classes)]
    doctor = ModelDoctor(model, class_names=class_names, num_classes=num_classes)
    report = doctor.diagnose(dataloader, n_clusters=2, n_example_images=3)

    assert report.misclassification.total_examples == 40
    assert 0.0 <= report.calibration.ece <= 1.0
    assert "<html" in report.html.lower()
    assert "Model Diagnosis Report" in report.html


def test_diagnose_auto_infers_num_classes(model, dataloader, num_classes):
    doctor = ModelDoctor(model)  # num_classes omitted on purpose
    assert doctor.num_classes == num_classes


def test_save_html_writes_file(model, dataloader):
    doctor = ModelDoctor(model)
    report = doctor.diagnose(dataloader, n_example_images=2)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.html")
        report.save_html(path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "<html" in content.lower()


def test_diagnose_handles_zero_example_images(model, dataloader):
    doctor = ModelDoctor(model)
    report = doctor.diagnose(dataloader, n_example_images=0)
    assert "<html" in report.html.lower()
