import numpy as np

from lensight import ConfidenceAnalyzer


def test_calibration_report_bins_sum_to_total(model, dataloader):
    analyzer = ConfidenceAnalyzer(model)
    report = analyzer.analyze(dataloader, n_bins=10)

    assert report.bin_counts.sum() == 40  # matches fixture dataset size
    assert report.n_bins == 10
    assert 0.0 <= report.ece <= 1.0
    assert 0.0 <= report.mce <= 1.0
    assert report.mce >= report.ece - 1e-9  # MCE is the max of the per-bin gaps


def test_calibration_report_summary_text(model, dataloader):
    analyzer = ConfidenceAnalyzer(model)
    report = analyzer.analyze(dataloader)
    text = report.summary_text()
    assert "ECE" in text


def test_perfectly_confident_correct_model_has_zero_ece():
    """A synthetic model that is always 100% confident and always correct
    should report ECE == 0 — a sanity check on the calibration math itself."""
    import torch
    import torch.nn as nn
    import torch.utils.data as data

    class AlwaysRightHugeLogit(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x):
            # x encodes the true label in x[:, 0, 0, 0] for this synthetic test.
            labels = x[:, 0, 0, 0].long().clamp(0, self.num_classes - 1)
            logits = torch.full((x.shape[0], self.num_classes), -1e4)
            logits[torch.arange(x.shape[0]), labels] = 1e4
            return logits

    labels = torch.randint(0, 3, (16,))
    images = torch.zeros(16, 3, 4, 4)
    images[:, 0, 0, 0] = labels.float()
    ds = data.TensorDataset(images, labels)
    loader = data.DataLoader(ds, batch_size=4)

    model = AlwaysRightHugeLogit()
    report = ConfidenceAnalyzer(model).analyze(loader, n_bins=10)

    assert np.isclose(report.overall_accuracy, 1.0)
    assert report.ece < 1e-6
