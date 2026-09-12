"""Self-contained interactive HTML report generator (zero external assets, zero external JS dependencies)."""

from __future__ import annotations
import base64
import html
import io
from datetime import datetime
from typing import Any

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from ..analysis.misclassification import MisclassificationReport
from ..analysis.confidence import CalibrationReport


def _image_to_data_uri(arr: np.ndarray) -> str:
    if Image is None:
        raise ImportError("Pillow is required to embed images in the HTML report.")
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _bar_svg(
    values: list[float],
    width: int = 440,
    height: int = 120,
    color: str = "#3b82f6",
    labels: list[str] | None = None,
) -> str:
    """Dependency-free bar chart as inline SVG for calibration diagrams."""
    if not values:
        return ""
    n = len(values)
    bar_w = width / n
    bars = []
    for i, v in enumerate(values):
        v = max(0.0, min(1.0, float(v)))
        bar_h = v * height
        x = i * bar_w
        y = height - bar_h
        bars.append(
            f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{max(1.0, bar_w - 4):.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" rx="2" class="bar" '
            f'data-val="{v:.3f}" data-bin="{i}">'
            f'<title>Bin {i+1}: {v:.1%}</title></rect>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width: {width}px;">'
        f'<line x1="0" y1="{height}" x2="{width}" y2="{height}" stroke="#cbd5e1" stroke-width="1.5"/>'
        f"{''.join(bars)}</svg>"
    )


def build_html_report(
    misclass_report: MisclassificationReport | None = None,
    calibration_report: CalibrationReport | None = None,
    example_images: list[Any] | None = None,
    title: str = "Model Diagnosis Report",
    eda_report: Any | None = None,
) -> str:
    """
    Assemble a single self-contained, interactive HTML report with zero external CDN dependencies.

    Parameters
    ----------
    misclass_report : MisclassificationReport, optional
    calibration_report : CalibrationReport, optional
    example_images : list of images, each either:
        - (caption, overlay_arr)
        - (caption, overlay_arr, meta_dict)
        - dict with 'caption', 'image', 'cluster', 'is_label_error', etc.
    title : str
    eda_report : EDAReport, optional
    """
    sections = []

    # Header
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections.append(
        f"""
        <header>
          <div class="header-content">
            <div class="header-badge">LENSIGHT DIAGNOSTIC DASHBOARD</div>
            <h1>{html.escape(title)}</h1>
            <p class="timestamp">Generated on {now_str} &bull; Self-contained report</p>
          </div>
        </header>
        """
    )

    # Data Quality / Suspected Label Errors Callout
    label_errors = []
    if misclass_report is not None:
        label_errors = misclass_report.find_label_errors(min_confidence=0.80)
        if label_errors:
            error_rows = "".join(
                f"<tr><td><code>#{err.index}</code></td>"
                f"<td><span class='badge badge-true'>{html.escape(misclass_report.label_name(err.true_label))}</span></td>"
                f"<td><span class='badge badge-pred'>{html.escape(misclass_report.label_name(err.pred_label))}</span></td>"
                f"<td><strong>{err.confidence:.1%}</strong></td>"
                f"<td>Cluster {err.cluster if err.cluster >= 0 else 'N/A'}</td></tr>"
                for err in label_errors[:8]
            )
            sections.append(
                f"""
                <section class="alert-section">
                  <div class="alert-header">
                    <span class="alert-icon">&#9888;</span>
                    <div>
                      <h2>Data Quality Alert: {len(label_errors)} Suspected Label Errors</h2>
                      <p>The model is highly confident (&ge; 80%) yet contradicts the dataset annotation. In real-world CV, these are overwhelmingly mislabeled ground-truth examples or ambiguous edge cases.</p>
                    </div>
                  </div>
                  <table class="interactive-table">
                    <thead><tr><th>Sample</th><th>Dataset Label</th><th>Model Prediction</th><th>Confidence</th><th>Failure Mode</th></tr></thead>
                    <tbody>{error_rows}</tbody>
                  </table>
                </section>
                """
            )

    # Accuracy & Failure Modes
    if misclass_report is not None:
        pairs_html = "".join(
            f'<tr class="filter-row" data-filter-pair="{t}_{p}">'
            f"<td><span class='badge badge-true'>{html.escape(misclass_report.label_name(t))}</span></td>"
            f"<td><span class='badge badge-pred'>{html.escape(misclass_report.label_name(p))}</span></td>"
            f"<td><strong>{c}</strong></td></tr>"
            for t, p, c in misclass_report.top_confused_pairs(8)
        )
        cluster_rows = "".join(
            f'<tr class="filter-row" data-filter-cluster="{cl["cluster"]}">'
            f"<td><strong>Cluster {cl['cluster']}</strong></td><td>{cl['size']} errors</td>"
            f"<td><span class='badge badge-true'>{html.escape(misclass_report.label_name(cl['dominant_true']))}</span> &rarr; "
            f"<span class='badge badge-pred'>{html.escape(misclass_report.label_name(cl['dominant_pred']))}</span></td>"
            f"<td>{cl['avg_confidence']:.1%}</td></tr>"
            for cl in misclass_report.cluster_summary()
        )
        sections.append(
            f"""
        <section>
          <h2>Accuracy &amp; Failure Modes</h2>
          <div class="stat-grid">
            <div class="stat-card">
              <span class="stat-value">{misclass_report.accuracy:.1%}</span>
              <span class="stat-label">Accuracy</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{misclass_report.total_errors}</span>
              <span class="stat-label">Errors / {misclass_report.total_examples}</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{misclass_report.n_clusters}</span>
              <span class="stat-label">Failure Clusters</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{len(label_errors)}</span>
              <span class="stat-label">Suspected Annotation Errors</span>
            </div>
          </div>

          <div class="split-tables">
            <div>
              <h3>Most Confused Class Pairs <small>(click to filter)</small></h3>
              <table class="interactive-table">
                <thead><tr><th>True Class</th><th>Predicted</th><th>Error Count</th></tr></thead>
                <tbody>{pairs_html or '<tr><td colspan="3">No errors found</td></tr>'}</tbody>
              </table>
            </div>
            <div>
              <h3>Failure Clusters <small>(click to filter)</small></h3>
              <table class="interactive-table">
                <thead><tr><th>Cluster</th><th>Size</th><th>Dominant Pattern</th><th>Avg Conf</th></tr></thead>
                <tbody>{cluster_rows or '<tr><td colspan="4">Not enough errors to cluster</td></tr>'}</tbody>
              </table>
            </div>
          </div>
        </section>
        """
        )

    # Confidence Calibration
    if calibration_report is not None:
        gap_bars = _bar_svg(
            list(np.abs(calibration_report.bin_confidence - calibration_report.bin_accuracy)),
            color="#ef4444",
        )
        acc_bars = _bar_svg(list(calibration_report.bin_accuracy), color="#3b82f6")

        remediation_html = ""
        if calibration_report.suggested_temperature is not None and calibration_report.calibrated_ece is not None:
            remediation_html = f"""
            <div class="remediation-box">
              <span class="remediation-tag">REMEDIATION AVAILABLE</span>
              <p>Applying Temperature Scaling (<strong>T = {calibration_report.suggested_temperature:.3f}</strong>) reduces ECE from <strong>{calibration_report.ece:.4f}</strong> to <strong>{calibration_report.calibrated_ece:.4f}</strong> without changing top-1 accuracy. Use <code>lensight.TemperatureScaler(model).fit(val_loader)</code> to calibrate.</p>
            </div>
            """

        sections.append(
            f"""
        <section>
          <h2>Confidence Calibration</h2>
          <div class="stat-grid">
            <div class="stat-card">
              <span class="stat-value">{calibration_report.ece:.4f}</span>
              <span class="stat-label">Expected Calibration Error (ECE)</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{calibration_report.mce:.4f}</span>
              <span class="stat-label">Max Calibration Error (MCE)</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{calibration_report.overall_confidence:.1%}</span>
              <span class="stat-label">Mean Model Confidence</span>
            </div>
          </div>
          <p class="summary-note">{html.escape(calibration_report.summary_text())}</p>
          {remediation_html}

          <div class="split-tables charts-row">
            <div class="chart-container">
              <h3>Per-Bin Empirical Accuracy (Low &rarr; High Confidence)</h3>
              {acc_bars}
            </div>
            <div class="chart-container">
              <h3>Calibration Gap |Confidence &minus; Accuracy|</h3>
              {gap_bars}
            </div>
          </div>
        </section>
        """
        )

    if eda_report is not None:
        alerts = []
        if eda_report.is_imbalanced:
            alerts.append(
                f"<div class='alert alert-warning'><strong>Class Imbalance:</strong> "
                f"Imbalance ratio is {eda_report.imbalance_ratio:.1f}x. Consider class weighting or oversampling.</div>"
            )
        if len(eda_report.duplicate_pairs) > 0:
            alerts.append(
                f"<div class='alert alert-danger'><strong>Duplicate Samples:</strong> "
                f"Found {len(eda_report.duplicate_pairs)} duplicate image pairs. Check for train/val split leakage.</div>"
            )
        if len(eda_report.underexposed_indices) > 0 or len(eda_report.overexposed_indices) > 0:
            alerts.append(
                f"<div class='alert alert-info'><strong>Exposure Outliers:</strong> "
                f"{len(eda_report.underexposed_indices)} underexposed (dark) and {len(eda_report.overexposed_indices)} overexposed (blown) images.</div>"
            )
        if len(eda_report.blurriest_indices) > 0:
            alerts.append(
                f"<div class='alert alert-info'><strong>Low Sharpness:</strong> "
                f"{len(eda_report.blurriest_indices)} images have low gradient energy (suspected motion blur or low focus).</div>"
            )

        alerts_html = "".join(alerts) or "<div class='alert alert-success'>All essential dataset health checks passed with no severe anomalies.</div>"

        sections.append(
            f"""
        <section>
          <h2>Dataset Exploratory Health (EDA)</h2>
          <div class="stat-grid">
            <div class="stat-card"><span class="stat-value">{eda_report.total_images}</span><span class="stat-label">Samples Profiled</span></div>
            <div class="stat-card"><span class="stat-value">{eda_report.imbalance_ratio:.1f}x</span><span class="stat-label">Imbalance Ratio</span></div>
            <div class="stat-card"><span class="stat-value">{eda_report.mean_brightness:.1%}</span><span class="stat-label">Mean Brightness</span></div>
            <div class="stat-card"><span class="stat-value">{len(eda_report.duplicate_pairs)}</span><span class="stat-label">Duplicate Pairs</span></div>
          </div>
          {alerts_html}
        </section>
        """
        )

    # Visual Explanations (Interactive Gallery)
    if example_images:
        cards = []
        cluster_set = set()
        for item in example_images:
            if isinstance(item, dict):
                cap = item.get("caption", "")
                arr = item.get("image", item.get("overlay"))
                cluster_id = item.get("cluster", -1)
                is_label_err = item.get("is_label_error", False)
                true_lbl = item.get("true_label", "")
                pred_lbl = item.get("pred_label", "")
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                cap, arr, meta = item
                cluster_id = meta.get("cluster", -1) if isinstance(meta, dict) else -1
                is_label_err = meta.get("is_label_error", False) if isinstance(meta, dict) else False
                true_lbl = meta.get("true_label", "") if isinstance(meta, dict) else ""
                pred_lbl = meta.get("pred_label", "") if isinstance(meta, dict) else ""
            else:
                cap, arr = item[0], item[1]
                cluster_id = -1
                is_label_err = False
                true_lbl = ""
                pred_lbl = ""

            if cluster_id >= 0:
                cluster_set.add(cluster_id)

            uri = _image_to_data_uri(arr)
            label_err_attr = "true" if is_label_err else "false"
            cards.append(
                f'<figure class="gallery-card" data-cluster="{cluster_id}" '
                f'data-label-error="{label_err_attr}" data-caption="{html.escape(cap.lower())}">'
                f'<div class="card-img-wrapper"><img src="{uri}" alt="{html.escape(cap)}"/></div>'
                f'<figcaption>{html.escape(cap)}</figcaption>'
                f'</figure>'
            )

        filter_buttons = ['<button class="chip active" data-filter="all">All Explanations</button>']
        if label_errors:
            filter_buttons.append('<button class="chip chip-alert" data-filter="label_error">&#9888; Suspected Label Errors</button>')
        for c in sorted(cluster_set):
            filter_buttons.append(f'<button class="chip" data-filter="cluster_{c}">Cluster {c}</button>')

        sections.append(
            f"""
        <section id="gallery-section">
          <div class="gallery-header">
            <h2>Visual Explanations (Grad-CAM Overlays)</h2>
            <div class="gallery-controls">
              <input type="text" id="gallery-search" placeholder="Search classes or captions..." />
            </div>
          </div>
          <div class="filter-chips">{''.join(filter_buttons)}</div>
          <div class="gallery" id="explanation-gallery">{''.join(cards)}</div>
        </section>
        """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{html.escape(title)}</title>
<link rel="icon" type="image/x-icon" href="favicon.ico"/>
<link rel="icon" type="image/png" sizes="32x32" href="favicon-32x32.png"/>
<link rel="icon" type="image/png" sizes="16x16" href="favicon-16x16.png"/>
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png"/>
<style>
  :root {{
    --bg-page: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --primary: #2563eb;
    --primary-light: #eff6ff;
    --border: #e2e8f0;
    --danger: #ef4444;
    --danger-bg: #fef2f2;
    --success: #10b981;
    --warning: #f59e0b;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background: var(--bg-page); color: var(--text-main); line-height: 1.5; }}
  header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 40px 48px; border-bottom: 1px solid #334155; }}
  .header-content {{ max-width: 1040px; margin: 0 auto; }}
  .header-badge {{ display: inline-block; font-size: 11px; letter-spacing: 0.1em; font-weight: 700; color: #60a5fa; background: rgba(96,165,250,0.15); padding: 4px 8px; border-radius: 4px; margin-bottom: 8px; }}
  header h1 {{ margin: 0; font-size: 28px; font-weight: 700; }}
  .timestamp {{ color: #94a3b8; margin: 8px 0 0; font-size: 13px; }}
  section {{ max-width: 1040px; margin: 28px auto; background: var(--card-bg); padding: 28px 36px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04); border: 1px solid var(--border); }}
  .alert-section {{ background: var(--danger-bg); border: 1px solid #fca5a5; }}
  .alert-header {{ display: flex; gap: 16px; align-items: flex-start; margin-bottom: 16px; }}
  .alert-icon {{ font-size: 28px; color: var(--danger); line-height: 1; }}
  .alert-header h2 {{ margin: 0 0 4px; font-size: 18px; color: #991b1b; }}
  .alert-header p {{ margin: 0; font-size: 13.5px; color: #7f1d1d; }}
  h2 {{ margin-top: 0; font-size: 20px; font-weight: 700; color: #0f172a; border-bottom: 1px solid var(--border); padding-bottom: 10px; }}
  h3 {{ font-size: 15px; color: #334155; margin: 0 0 10px; }}
  h3 small {{ font-weight: 400; color: var(--text-muted); font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; background: #f8fafc; }}
  .interactive-table tr.filter-row:hover, .interactive-table tbody tr:hover {{ background: var(--primary-light); cursor: pointer; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge-true {{ background: #e0e7ff; color: #3730a3; }}
  .badge-pred {{ background: #fee2e2; color: #991b1b; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: #f8fafc; border: 1px solid var(--border); padding: 16px 20px; border-radius: 8px; }}
  .stat-value {{ display: block; font-size: 30px; font-weight: 800; color: var(--primary); }}
  .stat-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em; }}
  .split-tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .chart-container {{ background: #f8fafc; border: 1px solid var(--border); padding: 16px; border-radius: 8px; }}
  .remediation-box {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 14px 18px; margin: 16px 0; color: #166534; font-size: 14px; }}
  .remediation-tag {{ font-size: 11px; font-weight: 800; color: #15803d; letter-spacing: 0.08em; display: block; margin-bottom: 4px; }}
  .remediation-box strong {{ color: #14532d; }}
  .summary-note {{ color: var(--text-muted); font-size: 14px; margin-bottom: 12px; }}
  .gallery-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .gallery-header h2 {{ border: none; padding: 0; margin: 0; }}
  #gallery-search {{ padding: 8px 14px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; width: 260px; }}
  #gallery-search:focus {{ outline: none; border-color: var(--primary); }}
  .filter-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
  .chip {{ background: #f1f5f9; border: 1px solid var(--border); padding: 6px 14px; border-radius: 20px; font-size: 12.5px; font-weight: 600; color: #334155; cursor: pointer; transition: all 0.15s ease; }}
  .chip:hover {{ background: #e2e8f0; }}
  .chip.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
  .chip-alert {{ background: #fee2e2; color: #991b1b; border-color: #fca5a5; }}
  .chip-alert.active {{ background: var(--danger); color: white; border-color: var(--danger); }}
  .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; }}
  .gallery-card {{ margin: 0; background: #ffffff; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: transform 0.15s ease, box-shadow 0.15s ease; }}
  .gallery-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
  .card-img-wrapper {{ width: 100%; aspect-ratio: 1; overflow: hidden; background: #000; }}
  .gallery-card img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  figcaption {{ font-size: 12px; color: #334155; padding: 10px 12px; font-weight: 500; border-top: 1px solid var(--border); }}
  @media (max-width: 768px) {{
    .split-tables {{ grid-template-columns: 1fr; }}
    section {{ padding: 20px; }}
    header {{ padding: 24px; }}
  }}
</style>
</head>
<body>
{''.join(sections)}

<script>
(function() {{
  const chips = document.querySelectorAll('.chip');
  const cards = document.querySelectorAll('.gallery-card');
  const searchInput = document.getElementById('gallery-search');
  let currentFilter = 'all';

  function applyFilter() {{
    const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
    cards.forEach(card => {{
      const cardCluster = card.getAttribute('data-cluster');
      const isLabelError = card.getAttribute('data-label-error') === 'true';
      const caption = card.getAttribute('data-caption') || '';

      let matchesFilter = false;
      if (currentFilter === 'all') {{
        matchesFilter = true;
      }} else if (currentFilter === 'label_error') {{
        matchesFilter = isLabelError;
      }} else if (currentFilter.startsWith('cluster_')) {{
        const targetC = currentFilter.replace('cluster_', '');
        matchesFilter = (cardCluster === targetC);
      }}

      let matchesSearch = (!query || caption.includes(query));
      card.style.display = (matchesFilter && matchesSearch) ? 'block' : 'none';
    }});
  }}

  chips.forEach(chip => {{
    chip.addEventListener('click', () => {{
      chips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      currentFilter = chip.getAttribute('data-filter');
      applyFilter();
    }});
  }});

  if (searchInput) {{
    searchInput.addEventListener('input', applyFilter);
  }}

  // Table row click-to-filter
  document.querySelectorAll('tr[data-filter-cluster]').forEach(row => {{
    row.addEventListener('click', () => {{
      const c = row.getAttribute('data-filter-cluster');
      const targetChip = document.querySelector(`.chip[data-filter="cluster_${{c}}"]`);
      if (targetChip) {{
        targetChip.click();
        const gallery = document.getElementById('gallery-section');
        if (gallery) gallery.scrollIntoView({{ behavior: 'smooth' }});
      }}
    }});
  }});
}})();
</script>
</body>
</html>
"""
