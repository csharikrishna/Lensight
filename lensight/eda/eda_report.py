"""Self-contained interactive HTML report generator for Dataset Exploratory Analysis & Auditing.

Supports:
  - Dataset EDA Reports (EDAReport)
  - Deep Dataset Audit Reports (AuditReport) with Duplicate Clusters,
    Train/Test Leakage, and Cross-Label Contradictions.
"""

from __future__ import annotations

import base64
import html
import io
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

if TYPE_CHECKING:
    from .auditor import AuditReport
    from .explorer import EDAReport


def _image_to_data_uri(arr: np.ndarray) -> str:
    if Image is None:
        raise ImportError("Pillow is required to embed images in the HTML report.")
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _distribution_svg(class_counts: dict[Any, int], label_fn=None, width: int = 500, height: int = 150) -> str:
    if not class_counts:
        return ""
    max_c = max(class_counts.values()) if class_counts else 1
    items = sorted(class_counts.items(), key=lambda x: str(x[0]))
    n = len(items)
    bar_w = width / max(1, n)
    bars = []

    for i, (cls_key, count) in enumerate(items):
        bar_h = (count / max_c) * (height - 24)
        x = i * bar_w
        y = (height - 24) - bar_h
        name = html.escape(str(label_fn(cls_key) if label_fn else cls_key))
        bars.append(
            f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{max(1.0, bar_w - 4):.1f}" '
            f'height="{bar_h:.1f}" fill="#3b82f6" rx="2">'
            f'<title>{name}: {count}</title></rect>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width: {width}px;">'
        f'<line x1="0" y1="{height - 24}" x2="{width}" y2="{height - 24}" stroke="#cbd5e1" stroke-width="1.5"/>'
        f"{''.join(bars)}"
        f'<text x="4" y="{height - 6}" font-size="11" fill="#64748b">Classes &rarr;</text>'
        f'</svg>'
    )


def build_eda_html_report(
    eda_report: EDAReport,
    title: str = "Dataset Health & Exploratory Data Analysis (EDA)",
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    dist_svg = _distribution_svg(eda_report.class_counts, eda_report.label_name)

    alerts = []
    if eda_report.is_imbalanced:
        alerts.append(
            f'<div class="alert alert-warning"><strong>Class Imbalance:</strong> '
            f'Imbalance ratio is {eda_report.imbalance_ratio:.1f}x. Consider class weighting or oversampling rare categories.</div>'
        )
    if len(eda_report.duplicate_pairs) > 0:
        alerts.append(
            f'<div class="alert alert-danger"><strong>Duplicate Samples Detected:</strong> '
            f'Found {len(eda_report.duplicate_pairs)} duplicate image pairs. Verify train/val split isolation to avoid leakage.</div>'
        )
    if len(eda_report.underexposed_indices) > 0 or len(eda_report.overexposed_indices) > 0:
        alerts.append(
            f'<div class="alert alert-info"><strong>Lighting Outliers:</strong> '
            f'{len(eda_report.underexposed_indices)} underexposed (dark) and {len(eda_report.overexposed_indices)} overexposed images detected.</div>'
        )
    if len(eda_report.blurriest_indices) > 0:
        alerts.append(
            f'<div class="alert alert-info"><strong>Low Sharpness:</strong> '
            f'{len(eda_report.blurriest_indices)} images have low gradient energy (suspected motion blur or low focus).</div>'
        )

    alerts_html = "".join(alerts) if alerts else '<div class="alert alert-success">All essential dataset health checks passed with no severe anomalies.</div>'

    cards = []
    for idx, tag, arr in eda_report.outlier_images:
        uri = _image_to_data_uri(arr)
        cards.append(
            f'<figure class="outlier-card">'
            f'<div class="img-wrap"><img src="{uri}" alt="{html.escape(tag)}"/></div>'
            f'<figcaption><strong>#{idx}</strong>: {html.escape(tag)}</figcaption>'
            f'</figure>'
        )

    gallery_section = ""
    if cards:
        gallery_section = f"""
        <section>
          <h2>Flagged Visual Outliers</h2>
          <p class="subtitle">Representative visual defects identified during data profiling:</p>
          <div class="gallery">{''.join(cards)}</div>
        </section>
        """

    class_rows = "".join(
        f"<tr><td><strong>{html.escape(eda_report.label_name(c))}</strong></td>"
        f"<td>{count}</td>"
        f"<td>{(count / eda_report.total_images) * 100:.1f}%</td></tr>"
        for c, count in sorted(eda_report.class_counts.items(), key=lambda x: -x[1])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --primary: #2563eb;
    --border: #e2e8f0;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: var(--bg); color: var(--text); line-height: 1.5; }}
  header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 36px 44px; border-bottom: 1px solid #334155; }}
  .header-content {{ max-width: 1040px; margin: 0 auto; }}
  .badge {{ display: inline-block; font-size: 11px; font-weight: 700; color: #60a5fa; background: rgba(96,165,250,0.15); padding: 4px 8px; border-radius: 4px; margin-bottom: 8px; }}
  header h1 {{ margin: 0; font-size: 26px; }}
  .timestamp {{ color: #94a3b8; font-size: 13px; margin: 6px 0 0; }}
  section {{ max-width: 1040px; margin: 24px auto; background: var(--card); padding: 26px 32px; border-radius: 12px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  h2 {{ margin-top: 0; font-size: 19px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  .subtitle {{ font-size: 13.5px; color: var(--muted); margin-top: -4px; margin-bottom: 16px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 16px; margin-bottom: 20px; }}
  .stat-card {{ background: #f8fafc; border: 1px solid var(--border); padding: 14px 18px; border-radius: 8px; }}
  .stat-value {{ display: block; font-size: 28px; font-weight: 800; color: var(--primary); }}
  .stat-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; font-weight: 600; }}
  .alert {{ padding: 12px 16px; border-radius: 8px; font-size: 13.5px; margin-bottom: 10px; }}
  .alert-danger {{ background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }}
  .alert-warning {{ background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }}
  .alert-info {{ background: #f0f9ff; border: 1px solid #bae6fd; color: #0369a1; }}
  .alert-success {{ background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 12px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; background: #f8fafc; }}
  .split {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 24px; }}
  .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; }}
  .outlier-card {{ margin: 0; background: white; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  .img-wrap {{ width: 100%; aspect-ratio: 1; background: #000; overflow: hidden; }}
  .outlier-card img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  figcaption {{ padding: 8px 10px; font-size: 12px; border-top: 1px solid var(--border); background: #f8fafc; }}
  @media (max-width: 768px) {{ .split {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <div class="header-content">
    <div class="badge">LENSIGHT DATASET PROFILER</div>
    <h1>{html.escape(title)}</h1>
    <p class="timestamp">Generated on {now_str} &bull; Plug-and-Play CV Exploratory Analysis</p>
  </div>
</header>

<section>
  <h2>Health Summary &amp; Alerts</h2>
  <div class="stat-grid">
    <div class="stat-card"><span class="stat-value">{eda_report.total_images}</span><span class="stat-label">Total Samples</span></div>
    <div class="stat-card"><span class="stat-value">{eda_report.imbalance_ratio:.1f}x</span><span class="stat-label">Imbalance Ratio</span></div>
    <div class="stat-card"><span class="stat-value">{eda_report.mean_brightness:.1%}</span><span class="stat-label">Mean Brightness</span></div>
    <div class="stat-card"><span class="stat-value">{len(eda_report.duplicate_pairs)}</span><span class="stat-label">Duplicate Pairs</span></div>
  </div>
  {alerts_html}
</section>

<section>
  <h2>Class Distribution &amp; Balance</h2>
  <div class="split">
    <div>
      <p class="subtitle">Sample counts per category:</p>
      <table>
        <thead><tr><th>Category</th><th>Samples</th><th>Frequency</th></tr></thead>
        <tbody>{class_rows or '<tr><td colspan="3">No class labels provided</td></tr>'}</tbody>
      </table>
    </div>
    <div>
      <p class="subtitle">Category representation chart:</p>
      {dist_svg}
    </div>
  </div>
</section>

{gallery_section}

</body>
</html>
"""


def build_audit_html_report(
    report: AuditReport,
    title: str = "Lensight Dataset Health & Audit Report",
) -> str:
    """Build a comprehensive, interactive HTML dashboard for AuditReport."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    dist_svg = _distribution_svg(report.class_counts)

    # Class rows
    class_rows = "".join(
        f"<tr><td><strong>{html.escape(str(c))}</strong></td>"
        f"<td>{count}</td>"
        f"<td>{(count / max(1, report.total_samples)) * 100:.1f}%</td></tr>"
        for c, count in sorted(report.class_counts.items(), key=lambda x: -x[1])
    )

    # Alerts
    alerts = []
    if report.imbalance_ratio >= 3.0:
        alerts.append(
            f'<div class="alert alert-warning"><strong>Severe Class Imbalance:</strong> '
            f'Imbalance ratio is {report.imbalance_ratio:.2f}x. Long-tail classes risk severe underfitting.</div>'
        )
    if report.duplicate_clusters:
        alerts.append(
            f'<div class="alert alert-warning"><strong>Duplicate Clusters Detected:</strong> '
            f'Found {len(report.duplicate_clusters)} clusters containing {report.total_duplicates_removed_count} redundant images.</div>'
        )
    if report.cross_label_conflicts:
        alerts.append(
            f'<div class="alert alert-danger"><strong>CRITICAL Cross-Label Contradictions:</strong> '
            f'Found {len(report.cross_label_conflicts)} duplicate pairs assigned to conflicting classes. Contradicts model supervision!</div>'
        )
    if report.leakage_matches:
        alerts.append(
            f'<div class="alert alert-danger"><strong>CRITICAL Train/Test Leakage:</strong> '
            f'Found {len(report.leakage_matches)} leaked test images that match training samples. Test accuracy is inflated!</div>'
        )
    if report.corrupt_files:
        alerts.append(
            f'<div class="alert alert-danger"><strong>Corrupt Files:</strong> '
            f'{len(report.corrupt_files)} file(s) failed to open.</div>'
        )

    alerts_html = "".join(alerts) if alerts else '<div class="alert alert-success">All data health and integrity checks passed! Dataset is clean.</div>'

    # Duplicate cluster cards
    id_to_sample = {s.sample_id: s for s in report.samples}
    cluster_cards = []
    for cl in report.duplicate_clusters[:30]:  # limit to 30 for HTML size
        surv_sample = id_to_sample.get(cl.survivor_id)
        surv_thumb = surv_sample.thumb_b64 if surv_sample else None
        surv_img_tag = f'<img src="{surv_thumb}" alt="Survivor"/>' if surv_thumb else '<div class="img-placeholder">Image</div>'
        surv_name = html.escape(str(cl.survivor_id))

        dup_blocks = []
        for dup_id in cl.duplicate_ids[:6]:
            dup_sample = id_to_sample.get(dup_id)
            dup_thumb = dup_sample.thumb_b64 if dup_sample else None
            dup_img_tag = f'<img src="{dup_thumb}" alt="Duplicate"/>' if dup_thumb else '<div class="img-placeholder">Image</div>'
            dup_dist = cl.distances_to_survivor.get(dup_id, 0)
            exact_tag = "EXACT" if dup_dist == 0 else f"Dist {dup_dist}"
            dup_name = html.escape(str(dup_id))
            dup_blocks.append(
                f'<div class="dup-item">'
                f'{dup_img_tag}'
                f'<div class="badge-removed">{exact_tag}</div>'
                f'<span class="subtext" title="{dup_name}">{os.path.basename(dup_name)}</span>'
                f'</div>'
            )

        cluster_cards.append(
            f'<div class="cluster-card" data-category="{html.escape(str(cl.category or ""))}">'
            f'<div class="cluster-header">'
            f'<span class="cluster-title">{html.escape(cl.cluster_id)}</span>'
            f'<span class="cluster-badge">{cl.total_count} items</span>'
            f'</div>'
            f'<div class="cluster-body">'
            f'<div class="survivor-item">'
            f'{surv_img_tag}'
            f'<div class="badge-kept">KEPT SURVIVOR</div>'
            f'<span class="subtext" title="{surv_name}">{os.path.basename(surv_name)}</span>'
            f'</div>'
            f'<div class="dup-list">{"".join(dup_blocks)}</div>'
            f'</div>'
            f'</div>'
        )

    # Cross-label cards
    cross_cards = []
    for conf in report.cross_label_conflicts[:20]:
        img_a = f'<img src="{conf.thumb_a}"/>' if conf.thumb_a else '<div class="img-placeholder">A</div>'
        img_b = f'<img src="{conf.thumb_b}"/>' if conf.thumb_b else '<div class="img-placeholder">B</div>'
        dist_str = "EXACT MATCH (dist=0)" if conf.is_exact else f"NEAR MATCH (dist={conf.hamming_distance})"
        cross_cards.append(
            f'<div class="pair-card">'
            f'<div class="pair-side">'
            f'<div class="pair-label class-a">{html.escape(conf.class_a)}</div>'
            f'{img_a}'
            f'<span class="subtext">{os.path.basename(str(conf.sample_a_id))}</span>'
            f'</div>'
            f'<div class="pair-middle">'
            f'<span class="vs-badge">VS</span>'
            f'<span class="dist-tag">{dist_str}</span>'
            f'</div>'
            f'<div class="pair-side">'
            f'<div class="pair-label class-b">{html.escape(conf.class_b)}</div>'
            f'{img_b}'
            f'<span class="subtext">{os.path.basename(str(conf.sample_b_id))}</span>'
            f'</div>'
            f'</div>'
        )

    # Leakage cards
    leak_cards = []
    for lk in report.leakage_matches[:20]:
        img_te = f'<img src="{lk.test_thumb}"/>' if lk.test_thumb else '<div class="img-placeholder">Test</div>'
        img_tr = f'<img src="{lk.train_thumb}"/>' if lk.train_thumb else '<div class="img-placeholder">Train</div>'
        dist_str = "EXACT MATCH (dist=0)" if lk.is_exact else f"NEAR MATCH (dist={lk.hamming_distance})"
        leak_cards.append(
            f'<div class="pair-card">'
            f'<div class="pair-side">'
            f'<div class="pair-label class-test">TEST ({html.escape(lk.test_class)})</div>'
            f'{img_te}'
            f'<span class="subtext">{os.path.basename(str(lk.test_id))}</span>'
            f'</div>'
            f'<div class="pair-middle">'
            f'<span class="leak-badge">&larr; LEAKED FROM</span>'
            f'<span class="dist-tag">{dist_str}</span>'
            f'</div>'
            f'<div class="pair-side">'
            f'<div class="pair-label class-train">TRAIN ({html.escape(lk.train_class)})</div>'
            f'{img_tr}'
            f'<span class="subtext">{os.path.basename(str(lk.train_id))}</span>'
            f'</div>'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #0b0f19;
    --surface: #111827;
    --surface-hover: #1f2937;
    --border: #374151;
    --text: #f9fafb;
    --text-muted: #9ca3af;
    --primary: #3b82f6;
    --danger: #ef4444;
    --warning: #f59e0b;
    --success: #10b981;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: var(--bg); color: var(--text); line-height: 1.5; }}
  header {{ background: linear-gradient(135deg, #111827 0%, #1e293b 100%); padding: 36px 40px; border-bottom: 1px solid var(--border); }}
  .header-container {{ max-width: 1140px; margin: 0 auto; }}
  .header-badge {{ display: inline-block; font-size: 11px; font-weight: 700; color: #60a5fa; background: rgba(59,130,246,0.18); padding: 4px 10px; border-radius: 4px; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }}
  header h1 {{ margin: 0; font-size: 26px; color: #fff; font-weight: 700; }}
  .timestamp {{ color: var(--text-muted); font-size: 13px; margin-top: 6px; }}

  main {{ max-width: 1140px; margin: 24px auto; padding: 0 16px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: var(--surface); border: 1px solid var(--border); padding: 16px 20px; border-radius: 10px; }}
  .stat-val {{ font-size: 28px; font-weight: 800; color: var(--primary); }}
  .stat-lbl {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; display: block; margin-top: 2px; }}

  .alert {{ padding: 14px 18px; border-radius: 8px; font-size: 14px; margin-bottom: 12px; }}
  .alert-danger {{ background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3); color: #fca5a5; }}
  .alert-warning {{ background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #fcd34d; }}
  .alert-success {{ background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #6ee7b7; }}

  .tabs {{ display: flex; gap: 8px; border-bottom: 1px solid var(--border); margin: 28px 0 20px; }}
  .tab-btn {{ background: transparent; border: none; color: var(--text-muted); padding: 10px 18px; font-size: 14px; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; }}
  .tab-btn.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
  .tab-pane {{ display: none; }}
  .tab-pane.active {{ display: block; }}

  .card-section {{ background: var(--surface); border: 1px solid var(--border); padding: 24px 28px; border-radius: 12px; margin-bottom: 24px; }}
  h2 {{ margin-top: 0; font-size: 18px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }}
  .subtext {{ font-size: 12px; color: var(--text-muted); display: block; max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  .split-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ color: var(--text-muted); text-transform: uppercase; font-size: 11.5px; }}

  /* Clusters */
  .cluster-card {{ background: #161f30; border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .cluster-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .cluster-title {{ font-weight: 700; color: #93c5fd; font-size: 14px; }}
  .cluster-badge {{ background: var(--surface); padding: 3px 8px; border-radius: 4px; font-size: 11px; color: var(--text-muted); }}
  .cluster-body {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  .survivor-item, .dup-item {{ text-align: center; width: 120px; }}
  .survivor-item img, .dup-item img {{ width: 110px; height: 110px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); background: #000; }}
  .img-placeholder {{ width: 110px; height: 110px; display: flex; align-items: center; justify-content: center; background: #222; border-radius: 6px; color: #666; font-size: 11px; }}
  .badge-kept {{ background: #065f46; color: #a7f3d0; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; margin: 4px 0; }}
  .badge-removed {{ background: #7f1d1d; color: #fecaca; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; margin: 4px 0; }}
  .dup-list {{ display: flex; gap: 12px; flex-wrap: wrap; }}

  /* Pairs */
  .pair-card {{ background: #161f30; border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-around; }}
  .pair-side {{ text-align: center; }}
  .pair-side img {{ width: 120px; height: 120px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); background: #000; }}
  .pair-label {{ font-size: 11.5px; font-weight: 700; padding: 3px 8px; border-radius: 4px; margin-bottom: 6px; }}
  .class-a {{ background: rgba(59,130,246,0.2); color: #93c5fd; }}
  .class-b {{ background: rgba(239,68,68,0.2); color: #fca5a5; }}
  .class-test {{ background: rgba(245,158,11,0.2); color: #fcd34d; }}
  .class-train {{ background: rgba(16,185,129,0.2); color: #6ee7b7; }}
  .pair-middle {{ text-align: center; }}
  .vs-badge, .leak-badge {{ font-size: 12px; font-weight: 800; color: #f59e0b; display: block; }}
  .dist-tag {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; display: block; }}
</style>
</head>
<body>
<header>
  <div class="header-container">
    <div class="header-badge">Lensight Dataset Health Suite</div>
    <h1>{html.escape(title)}</h1>
    <p class="timestamp">Audit generated on {now_str} &bull; Perceptual Hashing ({html.escape(report.algo.upper())}) &bull; Threshold &le; {report.threshold}</p>
  </div>
</header>

<main>
  <div class="stat-grid">
    <div class="stat-card"><span class="stat-val">{report.total_samples}</span><span class="stat-lbl">Audited Samples</span></div>
    <div class="stat-card"><span class="stat-val">{report.imbalance_ratio:.1f}x</span><span class="stat-lbl">Imbalance Ratio</span></div>
    <div class="stat-card"><span class="stat-val">{len(report.duplicate_clusters)}</span><span class="stat-lbl">Duplicate Clusters</span></div>
    <div class="stat-card"><span class="stat-val">{len(report.cross_label_conflicts)}</span><span class="stat-lbl">Cross-Label Errors</span></div>
    <div class="stat-card"><span class="stat-val">{len(report.leakage_matches)}</span><span class="stat-lbl">Leaked Test Images</span></div>
  </div>

  {alerts_html}

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-overview')">Overview &amp; Balance</button>
    <button class="tab-btn" onclick="switchTab('tab-duplicates')">Duplicate Clusters ({len(report.duplicate_clusters)})</button>
    <button class="tab-btn" onclick="switchTab('tab-crosslabel')">Cross-Label Conflicts ({len(report.cross_label_conflicts)})</button>
    <button class="tab-btn" onclick="switchTab('tab-leakage')">Train/Test Leakage ({len(report.leakage_matches)})</button>
  </div>

  <!-- TAB 1: OVERVIEW -->
  <div id="tab-overview" class="tab-pane active card-section">
    <h2>Category Distribution &amp; Dimensions</h2>
    <div class="split-grid">
      <div>
        <table>
          <thead><tr><th>Category</th><th>Count</th><th>Share</th></tr></thead>
          <tbody>{class_rows}</tbody>
        </table>
      </div>
      <div>
        <p style="color:var(--text-muted); font-size:13px; margin-top:0;">Category representation chart:</p>
        {dist_svg}
        <div style="margin-top:20px; font-size:13px; color:var(--text-muted);">
          <strong>Resolution Consistency:</strong><br/>
          Width range: {report.dimension_stats.get('width_min', 'N/A')} &ndash; {report.dimension_stats.get('width_max', 'N/A')} px<br/>
          Height range: {report.dimension_stats.get('height_min', 'N/A')} &ndash; {report.dimension_stats.get('height_max', 'N/A')} px<br/>
          Unique resolutions: {report.dimension_stats.get('unique_resolutions', 1)}
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: DUPLICATES -->
  <div id="tab-duplicates" class="tab-pane card-section">
    <h2>Duplicate Clusters (Transitive Union-Find)</h2>
    <p style="color:var(--text-muted); font-size:13px; margin-bottom:16px;">
      Redundant copies found within threshold &le; {report.threshold}. Lensight automatically selects the highest-sharpness sample as the Kept Survivor.
    </p>
    {"".join(cluster_cards) if cluster_cards else '<p style="color:#10b981;">[OK] No duplicate clusters found.</p>'}
  </div>

  <!-- TAB 3: CROSS-LABEL -->
  <div id="tab-crosslabel" class="tab-pane card-section">
    <h2>Cross-Label Contradictions</h2>
    <p style="color:var(--text-muted); font-size:13px; margin-bottom:16px;">
      Identical or near-duplicate images assigned to contradictory classes (e.g. Cat vs Dog). This injects contradictory loss gradients during training.
    </p>
    {"".join(cross_cards) if cross_cards else '<p style="color:#10b981;">[OK] No cross-label duplicate conflicts detected.</p>'}
  </div>

  <!-- TAB 4: LEAKAGE -->
  <div id="tab-leakage" class="tab-pane card-section">
    <h2>Train / Test Leakage Detection</h2>
    <p style="color:var(--text-muted); font-size:13px; margin-bottom:16px;">
      Test or validation images that match training samples. These artificially inflate evaluation performance.
    </p>
    {"".join(leak_cards) if leak_cards else '<p style="color:#10b981;">[OK] No train/test leakage detected.</p>'}
  </div>

</main>

<script>
function switchTab(tabId) {{
  document.querySelectorAll('.tab-btn').forEach(function(btn) {{ btn.classList.remove('active'); }});
  document.querySelectorAll('.tab-pane').forEach(function(pane) {{ pane.classList.remove('active'); }});
  if (event && event.target) {{ event.target.classList.add('active'); }}
  var targetPane = document.getElementById(tabId);
  if (targetPane) {{ targetPane.classList.add('active'); }}
}}
</script>
</body>
</html>
"""
