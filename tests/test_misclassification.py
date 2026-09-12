from lensight import MisclassificationAnalyzer


def test_analyze_returns_consistent_totals(model, dataloader, num_classes):
    analyzer = MisclassificationAnalyzer(model, num_classes=num_classes)
    report = analyzer.analyze(dataloader)

    assert report.total_examples == 40  # matches the fixture dataset size
    assert report.total_errors <= report.total_examples
    assert 0.0 <= report.accuracy <= 1.0
    assert report.confusion_matrix.sum() == report.total_examples
    assert report.confusion_matrix.shape == (num_classes, num_classes)


def test_samples_recorded_only_for_errors(model, dataloader, num_classes):
    analyzer = MisclassificationAnalyzer(model, num_classes=num_classes)
    report = analyzer.analyze(dataloader)

    assert len(report.samples) == report.total_errors
    for s in report.samples:
        assert s.true_label != s.pred_label


def test_clustering_assigns_valid_cluster_ids(model, dataloader, num_classes):
    analyzer = MisclassificationAnalyzer(model, num_classes=num_classes)
    report = analyzer.analyze(dataloader, n_clusters=3)

    if report.n_clusters > 0:
        for s in report.samples:
            assert 0 <= s.cluster < report.n_clusters


def test_top_confused_pairs_sorted_descending(model, dataloader, num_classes):
    analyzer = MisclassificationAnalyzer(model, num_classes=num_classes)
    report = analyzer.analyze(dataloader)

    pairs = report.top_confused_pairs(k=10)
    counts = [c for _, _, c in pairs]
    assert counts == sorted(counts, reverse=True)


def test_summary_text_runs_without_error(model, dataloader, num_classes):
    analyzer = MisclassificationAnalyzer(model, num_classes=num_classes)
    report = analyzer.analyze(dataloader)
    text = report.summary_text()
    assert "Accuracy" in text
