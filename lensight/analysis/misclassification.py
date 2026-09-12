"""
Misclassification analysis.

The recurring real-world debugging question this answers: "my model is at
92% accuracy — what's actually in the failing 8%, and is it random noise or
a systematic blind spot (one confused class pair, one lighting condition,
one object pose)?"

This module runs the model over a dataset, collects every misclassified
example together with a penultimate-layer embedding, and clusters those
embeddings so that visually/semantically similar failures group together —
turning a long flat list of wrong predictions into a handful of named
failure modes worth looking at.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..utils.model_utils import find_last_linear_layer, get_device


@dataclass
class MisclassifiedSample:
    index: int
    true_label: int
    pred_label: int
    confidence: float
    image: torch.Tensor  # the raw (preprocessed) input tensor, for re-explaining later
    cluster: int = -1


@dataclass
class MisclassificationReport:
    total_examples: int
    total_errors: int
    accuracy: float
    confusion_matrix: np.ndarray
    class_names: list[str] | None
    samples: list[MisclassifiedSample] = field(default_factory=list)
    n_clusters: int = 0
    suspected_label_errors: list[MisclassifiedSample] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return 1.0 - self.accuracy

    def find_label_errors(self, min_confidence: float = 0.80) -> list[MisclassifiedSample]:
        """
        Return misclassified samples where model confidence exceeded min_confidence.
        These high-confidence mistakes are primary candidates for label noise or
        ambiguous ground-truth annotations in the dataset.
        """
        return sorted(
            [s for s in self.samples if s.confidence >= min_confidence],
            key=lambda s: -s.confidence,
        )

    def top_confused_pairs(self, k: int = 5) -> list[tuple[int, int, int]]:
        """Return the (true, predicted, count) triples with the most confusions, off-diagonal."""
        cm = self.confusion_matrix.copy()
        np.fill_diagonal(cm, 0)
        flat_idx = np.argsort(cm, axis=None)[::-1]
        pairs = []
        for idx in flat_idx:
            true_i, pred_j = np.unravel_index(idx, cm.shape)
            count = cm[true_i, pred_j]
            if count == 0:
                break
            pairs.append((int(true_i), int(pred_j), int(count)))
            if len(pairs) == k:
                break
        return pairs

    def cluster_summary(self) -> list[dict]:
        """One entry per cluster: size, and the most common true/pred label pair in it."""
        summaries = []
        for c in range(self.n_clusters):
            members = [s for s in self.samples if s.cluster == c]
            if not members:
                continue
            true_labels = [s.true_label for s in members]
            pred_labels = [s.pred_label for s in members]
            mode_true = max(set(true_labels), key=true_labels.count)
            mode_pred = max(set(pred_labels), key=pred_labels.count)
            summaries.append(
                {
                    "cluster": c,
                    "size": len(members),
                    "dominant_true": mode_true,
                    "dominant_pred": mode_pred,
                    "avg_confidence": float(np.mean([s.confidence for s in members])),
                }
            )
        return sorted(summaries, key=lambda d: -d["size"])

    def label_name(self, idx: int) -> str:
        if self.class_names and 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)

    def summary_text(self) -> str:
        lines = [
            f"Accuracy: {self.accuracy:.2%} "
            f"({self.total_examples - self.total_errors}/{self.total_examples})",
            f"Errors analyzed: {self.total_errors}",
        ]
        pairs = self.top_confused_pairs(5)
        if pairs:
            lines.append("Most confused pairs (true -> predicted : count):")
            for t, p, c in pairs:
                lines.append(f"  {self.label_name(t)} -> {self.label_name(p)} : {c}")
        clusters = self.cluster_summary()
        if clusters:
            lines.append(f"Failure clusters found: {len(clusters)}")
            for cl in clusters[:5]:
                lines.append(
                    f"  cluster {cl['cluster']} (n={cl['size']}): mostly "
                    f"{self.label_name(cl['dominant_true'])} -> "
                    f"{self.label_name(cl['dominant_pred'])}, "
                    f"avg confidence {cl['avg_confidence']:.2f}"
                )
        label_errs = self.find_label_errors(0.80)
        if label_errs:
            lines.append(f"Suspected label errors (conf >= 80%): {len(label_errs)}")
            for err in label_errs[:3]:
                lines.append(
                    f"  idx {err.index}: true '{self.label_name(err.true_label)}' -> "
                    f"pred '{self.label_name(err.pred_label)}' ({err.confidence:.1%})"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_examples": self.total_examples,
            "total_errors": self.total_errors,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "top_confused_pairs": [
                {"true": self.label_name(t), "pred": self.label_name(p), "count": c}
                for t, p, c in self.top_confused_pairs()
            ],
            "n_clusters": self.n_clusters,
            "clusters": self.cluster_summary(),
            "suspected_label_errors": [
                {
                    "index": s.index,
                    "true_label": self.label_name(s.true_label),
                    "pred_label": self.label_name(s.pred_label),
                    "confidence": s.confidence,
                    "cluster": s.cluster,
                }
                for s in self.find_label_errors(0.80)
            ],
        }


class MisclassificationAnalyzer:
    """
    Usage
    -----
        analyzer = MisclassificationAnalyzer(model, num_classes=10, class_names=classes)
        report = analyzer.analyze(dataloader)
        print(report.summary_text())
        for sample in report.samples[:5]:
            ...  # re-run GradCAM on sample.image to see *why* it failed
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int | None = None,
        class_names: list[str] | None = None,
        embedding_layer: nn.Module | None = None,
        max_stored_samples: int = 500,
    ):
        self.model = model
        self.model.eval()
        if num_classes is None:
            if class_names is not None:
                num_classes = len(class_names)
            else:
                num_classes = find_last_linear_layer(model).out_features
        self.num_classes = num_classes
        self.class_names = class_names
        self.embedding_layer = embedding_layer or find_last_linear_layer(model)
        self.max_stored_samples = max_stored_samples
        self.device = get_device(model)

    def analyze(
        self,
        dataloader,
        n_clusters: int = 6,
        store_images: bool = True,
        label_error_threshold: float = 0.80,
    ) -> MisclassificationReport:
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=int)
        samples: list[MisclassifiedSample] = []
        embeddings: list[np.ndarray] = []

        captured = {}

        def _capture_input(module, inp):
            captured["embedding"] = inp[0].detach()

        handle = self.embedding_layer.register_forward_pre_hook(_capture_input)

        total = 0
        errors = 0
        try:
            with torch.no_grad():
                for batch in dataloader:
                    images, labels = batch[0], batch[1]
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(images)
                    probs = torch.softmax(logits, dim=1)
                    confs, preds = probs.max(dim=1)
                    batch_embeddings = captured["embedding"]

                    for i in range(images.shape[0]):
                        true_l = int(labels[i].item())
                        pred_l = int(preds[i].item())
                        confusion[true_l, pred_l] += 1
                        total += 1

                        if true_l != pred_l:
                            errors += 1
                            if len(samples) < self.max_stored_samples:
                                samples.append(
                                    MisclassifiedSample(
                                        index=total - 1,
                                        true_label=true_l,
                                        pred_label=pred_l,
                                        confidence=float(confs[i].item()),
                                        image=images[i].detach().cpu()
                                        if store_images
                                        else torch.empty(0),
                                    )
                                )
                                embeddings.append(
                                    batch_embeddings[i].detach().cpu().numpy()
                                )
        finally:
            handle.remove()

        accuracy = (total - errors) / total if total else 0.0

        n_clusters_used = self._assign_clusters(samples, embeddings, n_clusters)
        suspected_errors = [s for s in samples if s.confidence >= label_error_threshold]

        return MisclassificationReport(
            total_examples=total,
            total_errors=errors,
            accuracy=accuracy,
            confusion_matrix=confusion,
            class_names=self.class_names,
            samples=samples,
            n_clusters=n_clusters_used,
            suspected_label_errors=suspected_errors,
        )

    @staticmethod
    def _assign_clusters(samples, embeddings, n_clusters) -> int:
        if len(samples) < 2:
            return 0
        n_clusters = max(1, min(n_clusters, len(samples)))
        try:
            from sklearn.cluster import KMeans

            X = np.stack(embeddings)
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = km.fit_predict(X)
            for sample, cluster_id in zip(samples, labels):
                sample.cluster = int(cluster_id)
            return n_clusters
        except ImportError:
            return 0
