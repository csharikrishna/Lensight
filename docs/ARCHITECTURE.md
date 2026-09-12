# Lensight System Architecture

This document provides an in-depth technical explanation of the internal architecture, mathematical formulations, and engineering designs powering **Lensight**.

---

## 1. High-Level Design Philosophy

Lensight is built on four architectural principles:

1. **Unified Vision Lifecycle**: Data Quality (Pre-Training) and Model Quality (Post-Training) form a single closed-loop debugging cycle. They belong in the same unified toolkit.
2. **Zero-Bloat Dependency Diet**: Core algorithms are strictly built on **PyTorch**, **NumPy**, **Pillow**, and **Scikit-Learn**. No OpenCV, SciPy, Pandas, or heavy C++ extensions are required.
3. **Non-Destructive In-Memory Operations**: Datasets are never modified or duplicated on disk by default. Transformations, clean subsets, and calibrations operate in-memory using native PyTorch abstractions (`torch.utils.data.Subset`, `torch.nn.Module`).
4. **Vectorized Computation**: All distance matrices, perceptual hashing comparisons, and batch attributions are vectorized in C/AVX registers via NumPy and PyTorch tensor operations, avoiding nested Python loop overhead.

---

## 2. Architectural Blueprint

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 LENSIGHT UNIFIED ENGINE                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [ PRE-TRAINING DATA HEALTH ]                    [ POST-TRAINING MODEL DIAGNOSTICS ]
 ┌───────────────────────────┐                   ┌───────────────────────────────────┐
 │   lensight.eda.auditor    │                   │       lensight.cam & saliency     │
 │   • dHash / aHash / pHash │                   │       • GradCAM / GradCAM++       │
 │   • Transitive Union-Find │                   │       • HiResCAM / ContrastiveCAM │
 │   • Smart Survivor (Blur) │                   │       • IntegratedGradients       │
 │   • Cross-Label Conflicts │                   │       • SmoothGrad                │
 └─────────────┬─────────────┘                   └─────────────────┬─────────────────┘
               │                                                   │
 ┌─────────────▼─────────────┐                   ┌─────────────────▼─────────────────┐
 │   lensight.eda.sanitizer  │                   │     lensight.analysis.confidence  │
 │   • Non-destructive Subset│                   │     • ECE / MCE Reliability Bins  │
 │   • In-memory clean index │                   │     • TemperatureScaler (L-BFGS)  │
 └─────────────┬─────────────┘                   └─────────────────┬─────────────────┘
               │                                                   │
               │   [PyTorch DataLoader]                            │
               └───────────────► ┌───────────────────┐ ◄───────────┘
                                 │    ModelDoctor    │
                                 │ (One-Shot Orchestr)│
                                 └─────────┬─────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
             ┌──────────▼──────────┐               ┌──────────▼──────────┐
             │ Interactive HTML UI │               │ Machine-Readable    │
             │ (Self-contained JS) │               │ JSON Metrics        │
             └─────────────────────┘               └─────────────────────┘
```

---

## 3. Module Breakdown & Mechanics

### Pillar 1: Pre-Training Data Quality (`lensight.eda`)

#### Vectorized Perceptual Hashing & Fast Distance Matrices
Legacy tools (`imgcheck`, `imagehash`) compute pairwise Hamming distances using nested Python loops:

$$\mathcal{O}\left(\frac{N(N-1)}{2}\right) \text{ individual Python function calls}$$

Lensight replaces this with **vectorized bit-unpacking and a 256-element popcount lookup table**:
1. 64-bit difference hashes are stored as 8-byte uint8 vectors (`shape: (N, 8)`).
2. Given two sets of hashes $A \in \mathbb{R}^{M \times 8}$ and $B \in \mathbb{R}^{N \times 8}$, the pairwise XOR matrix is computed in one vectorized operation:
   $$\text{Diff} = A[:, \text{None}, :] \oplus B[\text{None}, :, :] \quad \in \mathbb{R}^{M \times N \times 8}$$
3. The population count (number of set bits) is resolved via a constant-time precomputed 256-element integer table:
   $$\text{DistanceMatrix} = \sum_{k=0}^{7} \text{PopcountTable}\left[\text{Diff}[:, :, k]\right]$$
4. This executes at native C/SIMD speed, analyzing thousands of image pairs in milliseconds.

#### Transitive Duplicate Clustering with Smart Survivor Selection
Given a Hamming distance threshold $\tau$:
1. A disjoint-set (Union-Find) with path compression clusters transitively connected images ($A \sim B$ and $B \sim C \implies \{A, B, C\}$).
2. **Smart Survivor Selection**: Rather than keeping an arbitrary image (e.g. alphabetical file path as in legacy tools), Lensight computes an image quality metric $Q(x)$ for each member:
   $$Q(x) = \text{Var}\left(\nabla^2 I_{\text{gray}}\right) \times \left(W \times H\right)$$
   Where $\nabla^2$ is the discrete 2D Laplacian operator:
   $$\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
   The candidate with the highest $Q(x)$ is chosen as the **survivor**, guaranteeing that high-resolution, sharp master copies are preserved while blurry or corrupted duplicates are flagged for removal.

#### In-Memory Sanitization (`DatasetSanitizer`)
Instead of duplicating tens of gigabytes of image files into new directories on disk, `DatasetSanitizer.clean_subset` constructs a filtered index set:

$$\mathcal{I}_{\text{clean}} = \{0, \dots, N-1\} \setminus (\mathcal{I}_{\text{duplicates}} \cup \mathcal{I}_{\text{cross-conflict}} \cup \mathcal{I}_{\text{corrupt}})$$

It instantiates a clean `torch.utils.data.Subset(dataset, indices=list(clean_indices))` with zero disk mutation.

---

### Pillar 2: Explainability & Attribution (`lensight.cam` & `lensight.saliency`)

#### Hook Architecture & Lifecycle Management
Explainers attach forward and backward hooks to target PyTorch layers via `ActivationsAndGradients` (`lensight/utils/hooks.py`):
1. **Forward Hook**: Captures intermediate activations $A^k \in \mathbb{R}^{C \times H \times W}$ during standard inference.
2. **Backward Hook**: Intercepts gradients $\frac{\partial Y^c}{\partial A^k}$ flowing back from target class logit $Y^c$.
3. **Context Management**: Hooks are cleanly registered and automatically deregistered to prevent memory leaks in long-running training loops.

#### Vectorized Batch CAM Computation
Standard CAM libraries loop over each batch item one-by-one, requiring $B$ separate backward passes. Lensight implements batched execution in `BaseCAM.explain_batch`:
1. Vectorized forward pass produces logits for batch $B$.
2. Target class scores are summed across the batch:
   $$S = \sum_{i=1}^B Y_{i}^{c_i}$$
3. A single `loss.backward()` call computes exact, independent gradients for all $B$ samples simultaneously:
   $$\frac{\partial S}{\partial A_i^k} = \frac{\partial Y_i^{c_i}}{\partial A_i^k}$$
4. Speedup is directly proportional to GPU batch parallelism ($4\times$ to $16\times$ throughput increase).

#### Contrastive CAM Mechanics
Answers: *"Why did the network predict Class $A$ instead of Class $B$?"*
$$L_{\text{contrast}} = Y^{\text{target}} - Y^{\text{contrast}}$$
By backpropagating $L_{\text{contrast}}$, channels that support $A$ receive positive weights, while channels that support $B$ receive negative weights. Positive rectification yields only the visual features that uniquely distinguish $A$ from $B$.

---

### Pillar 3: Confidence Calibration & Uncertainty (`lensight.analysis.confidence`)

#### Expected Calibration Error (ECE)
Samples are partitioned into $M$ equally-spaced confidence bins $B_1, \dots, B_M \subset (0, 1]$:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

$$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

#### Temperature Scaling Remediation
Temperature Scaling (Guo et al., ICML 2017) rescales unnormalized model logits $z_i$ by a single scalar temperature $T > 0$:

$$\hat{p}_i = \max_{k} \sigma\left(\frac{z_i}{T}\right)_k$$

* Parameter $T$ is optimized on validation set logits using L-BFGS to minimize Negative Log-Likelihood (NLL).
* **Mathematical Invariance**: Because temperature scaling is a strictly monotonic transformation for any $T > 0$:
  $$\arg\max_k \sigma\left(\frac{z}{T}\right)_k = \arg\max_k \sigma(z)_k$$
* The model's top-1 accuracy is **100% invariant**, while the probability distribution becomes accurately calibrated.

---

### Pillar 4: Error Profiling & Semantic Clustering (`lensight.analysis.misclassification`)

1. For each misclassified sample $(x_i, y_i, \hat{y}_i)$, Lensight extracts the penultimate feature embedding $e_i \in \mathbb{R}^D$ (input to the final classification layer).
2. Embeddings are projected via PCA to reduce noise and clustered using $k$-Means.
3. Each cluster surfaces a distinct failure mode (e.g. Cluster 1: *High-exposure background confusion*, Cluster 2: *Specific class pair confusion*).
4. **Suspected Label Error Identification**: Samples where model confidence $p(\hat{y}_i | x_i) \ge \theta_{\text{clean}}$ (default 80%) while disagreeing with the dataset label $y_i$ are flagged as candidate label errors.

---

### Pillar 5: Unified Orchestration & Dashboards (`lensight.doctor` & `lensight.report`)

* [`ModelDoctor`](file:///c:/Users/cshar/Downloads/lensight/lensight/doctor.py) serves as the top-level facade coordinating data loading, XAI attribution, error clustering, and calibration analysis in a single method call: `doctor.diagnose(dataloader)`.
* **Zero-Dependency HTML Reporting**:
  * Emits single-file, self-contained HTML reports with zero CDN dependencies.
  * Images are encoded as base64 data URIs.
  * Charts are rendered as responsive inline SVGs.
  * Filter chips, search bars, and tab navigation run on vanilla JavaScript with zero external scripts.
  * Supports notebook embedding via the `_repr_html_()` protocol.

---

## 4. Tensor Shape Conventions

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `GradCAM.explain` | `(C, H, W)` or `(1, C, H, W)` | `(h, w)` of target layer | Values normalized to `[0.0, 1.0]` |
| `overlay_heatmap` | `(H, W, 3)` image, `(h, w)` map | `(H, W, 3)` uint8 image | Bilinear resampling to match image size |
| `IntegratedGradients` | `(C, H, W)` | `(H, W)` | Absolute attribution aggregated across channels |
| `DatasetAuditor.audit` | `DataLoader` / folder path | `AuditReport` dataclass | Contains clusters, conflicts, and dimension stats |
| `TemperatureScaler.fit` | Validation `DataLoader` | `CalibrationSummary` | Optimizes scalar $T \in \mathbb{R}^+$ |
