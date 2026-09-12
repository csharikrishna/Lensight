# API Reference

## `lensight.cam`

### `GradCAM(model, target_layer=None, reshape_transform=None)`
- `target_layer`: an `nn.Module` inside `model` to explain. If omitted, the
  last `nn.Conv2d` in the model (by execution order) is used automatically.
- `reshape_transform`: optional callable `(tensor) -> tensor` used to reshape
  Vision Transformer patch tokens `(B, N, D)` to spatial maps `(B, D, H, W)`.
  See `lensight.utils.model_utils.create_vit_reshape_transform`.
- `.explain(input_tensor, target_class=None) -> np.ndarray`
  - `input_tensor`: shape `(C, H, W)` or `(1, C, H, W)`.
  - `target_class`: class index to explain; defaults to the model's own
    predicted class (`argmax` of the output logits).
  - Returns a `(h, w)` float array in `[0, 1]`.
- `.explain_batch(input_batch, target_classes=None) -> list[np.ndarray]`
  - Computes heatmaps for a batch of images using a single vectorized forward
    and backward pass for high GPU throughput.

### `GradCAMPlusPlus(model, target_layer=None, reshape_transform=None)`
Same interface as `GradCAM`. Prefer this when Grad-CAM's heatmap looks too
diffuse, or the image contains multiple instances of the target class.

### `HiResCAM(model, target_layer=None, reshape_transform=None)`
Draelos & Carin, NeurIPS 2020. Eliminates spatial average pooling before
weighting, computing elementwise gradients:
$$L_{\text{HiResCAM}} = \text{ReLU}\left(\sum_k \frac{\partial Y^c}{\partial A^k} \odot A^k\right)$$
Guarantees higher fidelity for fine-grained localization.

### `ContrastiveCAM(model, target_layer=None, reshape_transform=None)`
Answers: *"Why class A instead of class B?"*
- `.explain(input_tensor, target_class=None, contrast_class=None) -> np.ndarray`
  - If `contrast_class` is omitted, automatically selects the runner-up class.
  - Backpropagates $z_{\text{target}} - z_{\text{contrast}}$ to isolate features
    that favored the target over the competitor.

### `EigenCAM(model, target_layer=None, reshape_transform=None)`
Gradient-free CAM via principal components (SVD). Useful when gradients
are uninformative, clipped, or unavailable.

## `lensight.saliency`

### `VanillaGradient(model)`
- `.explain(input_tensor, target_class=None) -> np.ndarray`, same shape as
  the input's spatial dimensions `(H, W)`.

### `IntegratedGradients(model, steps=50)`
- `.explain(input_tensor, target_class=None, baseline=None) -> np.ndarray`
  - `baseline`: reference input to integrate from; defaults to an all-zero
    tensor.

### `SmoothGrad(model, n_samples=25, noise_level=0.15)`
- `.explain(input_tensor, target_class=None) -> np.ndarray`
  - `noise_level`: standard deviation of Gaussian noise added per sample.

## `lensight.analysis`

### `MisclassificationAnalyzer(model, num_classes, class_names=None, embedding_layer=None, max_stored_samples=500)`
- `.analyze(dataloader, n_clusters=6, store_images=True, label_error_threshold=0.80) -> MisclassificationReport`
  - Returns a `MisclassificationReport` with:
    - `.accuracy`, `.total_examples`, `.total_errors`
    - `.confusion_matrix`: `(num_classes, num_classes)` numpy array
    - `.samples`: list of `MisclassifiedSample`
    - `.top_confused_pairs(k=5)`
    - `.cluster_summary()`
    - `.find_label_errors(min_confidence=0.80)`: returns high-confidence mistakes indicating dataset annotation noise
    - `.to_dict()`: serializable dictionary for MLOps tracking
    - `.summary_text()`

### `ConfidenceAnalyzer(model)`
- `.analyze(dataloader, n_bins=15) -> CalibrationReport`
  - Returns a `CalibrationReport` with `.ece`, `.mce`, per-bin arrays,
    `.suggested_temperature`, `.calibrated_ece`, `.to_dict()`, and `.summary_text()`.

### `TemperatureScaler(model)`
Guo et al., ICML 2017. Post-hoc probability calibration.
- `.fit(dataloader, lr=0.01, max_iter=50, n_bins=15) -> CalibrationSummary`
  - Optimizes scalar temperature $T > 0$ on validation logits via NLL loss.
  - Returns a `CalibrationSummary` with `temperature`, `initial_ece`, `calibrated_ece`,
    `ece_reduction`, `initial_nll`, `calibrated_nll`, and `.to_dict()`.
- `.calibrated_model -> CalibratedModel`
  - Returns an `nn.Module` wrapper that divides logits by $T$ and provides
    `predict_proba(x)`. Preserves original top-1 classification accuracy.

## `lensight.eda`

### `DatasetExplorer(class_names=None, normalize_mean=None, normalize_std=None, max_samples=None, random_seed=42)`
Plug-and-play exploratory data analysis for computer vision datasets. Evaluates 5 fundamental CV dataset health signals without large memory overhead:
1. **Class distribution & imbalance ratio**: Max-to-min class ratio with Gini coefficient.
2. **Exposure & contrast health**: Highlights underexposed (dark clipping), overexposed (saturated highlights), and flat low-contrast images.
3. **Sharpness & blur detection**: 3x3 Laplacian filter variance estimation identifying degraded, out-of-focus, or motion-blurred inputs.
4. **Dimension & aspect ratio consistency**: Checks for irregular dimensions and varying aspect ratios.
5. **Exact & perceptual duplicate detection**: 64-bit difference hashing (`dhash`) identifying identical or near-identical sample pairs (e.g. data leakage between train and test splits).

- `.analyze(dataset_or_loader) -> EDAReport`
  - Accepts a PyTorch `DataLoader`, `Dataset`, or list of `(image, label)` tuples.
  - Streaming computation: accumulates running stats and perceptual hashes on-the-fly.

### `EDAReport`
Structured result object returned by `DatasetExplorer.analyze(...)`.
- Attributes:
  - `total_samples`: total number of processed images.
  - `class_counts`: dict mapping class name to count.
  - `imbalance_ratio`: float (ratio of majority to minority class count).
  - `exposure_stats`: dict with mean, std, dark/bright/low-contrast outlier lists.
  - `sharpness_stats`: dict with mean, std, blur threshold, and blur outlier list.
  - `dimension_stats`: dict with unique shapes and count of irregular sizes.
  - `duplicate_pairs`: list of duplicate sample pairs with Hamming distance.
  - `warnings`: list of actionable strings indicating discovered data anomalies.
- Methods:
  - `.summary_text() -> str`: returns clean ASCII summary for terminal logs.
  - `.to_dict() -> dict`: JSON-serializable dictionary for automated data tests / MLOps telemetry.
  - `.save_html(path)`: exports a standalone, dark-themed interactive HTML report with embedded outlier galleries and SVG distribution charts.
  - `._repr_html_()`: renders interactive HTML directly inside Jupyter and Colab notebooks.

### `DatasetAuditor(class_names=None, max_samples=None, store_thumbnails=True)`
Deep multi-split dataset auditing engine supporting both PyTorch loaders/datasets and disk directories (`folder_root/<class>/<images>`).
- `.audit(data_source, threshold=3, algo='dhash') -> AuditReport`
  - Runs transitive Union-Find duplicate clustering with smart survivor selection (highest Laplacian sharpness).
  - Detects contradictory cross-label duplicate images across classes.
  - Detects corrupt, truncated, or unreadable image files.
- `.check_leakage(train_source, test_source, threshold=3, algo='dhash') -> LeakageReport`
  - Cross-references test images against train images of the same category.
  - Reports leaked sample count, leakage percentage per class, and exact/near matches.

### `AuditReport`
- Attributes:
  - `total_samples`: int
  - `class_counts`: dict mapping class to count
  - `imbalance_ratio`: float
  - `corrupt_files`: list of `CorruptFileInfo`
  - `dimension_stats`: dict with min/max dimensions and varying resolution flags
  - `duplicate_clusters`: list of `DuplicateGroup` with transitive clusters, survivor ID, and duplicate IDs
  - `cross_label_conflicts`: list of `CrossLabelConflict`
  - `leakage_matches`: list of `LeakageMatch`
- Methods:
  - `.get_clean_indices(remove_duplicates=True, remove_cross_label=True) -> list[int]`: clean integer indices for `torch.utils.data.Subset`.
  - `.export_clean_dataset(output_dir, mode='copy') -> int`: copies clean survivors to target directory mirroring class subfolders.
  - `.export_manifest(path)`: exports clean JSON audit manifest.
  - `.save_html(path)`: exports dark-themed interactive HTML report with tabs for Clusters, Cross-Label conflicts, and Leakage.
  - `._repr_html_()`: native notebook rendering.

### `DatasetSanitizer`
Remediation utilities:
- `DatasetSanitizer.clean_subset(dataset, audit_report) -> torch.utils.data.Subset`: creates an in-memory cleaned PyTorch dataset with zero disk mutation.
- `DatasetSanitizer.clean_test_subset(test_dataset, leakage_report) -> torch.utils.data.Subset`: creates an in-memory leakage-free test dataset.
- `DatasetSanitizer.export_clean_directory(source_dir, output_dir, audit_report, mode='copy')`: exports clean files to disk.
- `DatasetSanitizer.clean_inplace(source_dir, audit_report, confirm=True)`: controlled in-place deletion requiring explicit confirmation.

### CLI Reference (`lensight`)
Installed CLI or runnable via `python -m lensight.cli`:
- `lensight audit --dataset <path> [--report audit.html] [--json metrics.json] [--threshold 3]`
- `lensight leakage --train <train_path> --test <test_path> [--threshold 3] [--json metrics.json]`
- `lensight clean --dataset <path> --output <clean_path> [--mode copy|inplace] [--confirm]`

## `lensight.ModelDoctor`

### `ModelDoctor(model, class_names=None, num_classes=None, normalize_mean=..., normalize_std=...)`
- `.diagnose(dataloader, n_clusters=6, n_example_images=6, cam_target_layer=None, report_title=..., compute_calibration_remediation=True, include_eda=False) -> DiagnosisReport`
  - Orchestrates failure mode clustering, calibration analysis, temperature scaling remediation,
    visual overlays, and optional EDA profiling into a single call.
  - `include_eda`: boolean (default `False`). When `True`, automatically runs `DatasetExplorer` on the validation loader and attaches dataset health findings directly into the diagnostic report.
  - Returns a `DiagnosisReport` with:
    - `.misclassification`, `.calibration`, `.eda_report`, `.html`, `.summary_text()`
    - `.save_html(path)`: writes self-contained interactive dashboard (includes EDA section if `include_eda=True`)
    - `._repr_html_()`: renders interactive dashboard natively in Jupyter, Colab, and VS Code notebooks
    - `.to_dict()`, `.to_json(path=None)`: export metrics to CI/CD and MLOps platforms

## `lensight.utils.model_utils`

- `create_vit_reshape_transform(has_cls_token=True)`: adapter creating a reshape transform for Vision Transformers (ViT, DeiT, DINO, etc.).
- `find_last_conv_layer(model)`: locates the last `nn.Conv2d` layer.
- `find_target_layer(model)`: locates best candidate layer for CAM (Conv2d, LayerNorm, or feature block).
- `find_last_linear_layer(model)`: locates penultimate embedding layer.
- `get_device(model)`: returns the model's parameter device.
- `named_module(model, dotted_name)`: fetches submodule by dotted string path.

## `lensight.utils.image_utils`

- `denormalize(tensor, mean=..., std=...) -> np.ndarray`: CHW tensor → HWC uint8 image.
- `overlay_heatmap(image, heatmap, alpha=0.45) -> np.ndarray`: blends heatmap onto RGB image.
- `apply_colormap(heatmap) -> np.ndarray`: colorizes float heatmap in `[0, 1]`.
