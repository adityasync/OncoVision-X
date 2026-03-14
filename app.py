#!/usr/bin/env python3
"""
OncoVision-X Web Demo — Flask Backend

Complete web backend with:
  - CT scan upload and analysis (/api/analyze)
  - CT slice visualization with marked nodules (/api/visualize)
  - Demo mode with simulated results (/api/demo-analyze)
  - Health check (/health)

Usage:
  python app.py                     # Start on port 5000
  python app.py --port 8080         # Custom port
  python app.py --demo              # Demo-only mode (no models required)

NOT part of the research paper — demo feature only.
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Lazy-loaded global system ──
_system = None
_demo_only = False


def get_system():
    """Lazy-load the detection + classification system."""
    global _system
    if _system is not None:
        return _system
    if _demo_only:
        return None

    from src.pipeline.end_to_end import LungCancerDetectionSystem

    detection_ckpt = os.environ.get(
        'DETECTION_CHECKPOINT', 'experiments/full_model/checkpoints/best.pth')
    classifier_ckpt = os.environ.get(
        'CLASSIFIER_CHECKPOINT', 'pretrained/resnet_18_23dataset.pth')
    detection_config = os.environ.get(
        'DETECTION_CONFIG', 'configs/full_model.yaml')

    _system = LungCancerDetectionSystem(
        detection_model_path=detection_ckpt if Path(detection_ckpt).exists() else None,
        classifier_model_path=classifier_ckpt if Path(classifier_ckpt).exists() else None,
        detection_config_path=detection_config,
    )
    return _system


# ── Visualization ──

RISK_COLORS = {
    'high_risk': '#dc3545',
    'medium_risk': '#ffc107',
    'low_risk': '#28a745',
    'uncertain': '#6c757d',
}


def _nodule_risk_style(nodule):
    """Dual-factor risk: malignancy probability + detection confidence."""
    mal = nodule.get('malignancy_probability', 0.5)
    conf = nodule.get('detection_confidence', 0.5)

    if conf < 0.60:
        return 'uncertain', RISK_COLORS['uncertain'], '--'
    if mal > 0.70:
        return 'high_risk', RISK_COLORS['high_risk'], '-'
    elif mal > 0.40:
        return 'medium_risk', RISK_COLORS['medium_risk'], '-'
    else:
        return 'low_risk', RISK_COLORS['low_risk'], '-'


def create_visualization(ct_scan, nodules, patient_id='Unknown'):
    """Professional 3-panel PACS-style medical imaging visualization.

    Layout:
      Row 0: Axial (main) | Coronal | Sagittal   — all nodules marked
      Row 1: Patient summary bar
      Row 2: Per-nodule clinical detail text

    Returns:
        Base64-encoded PNG string
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(20, 14), facecolor='#0a0a0a')
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25,
                          height_ratios=[3, 0.8, 2])

    # ── Best axial slice (most nodules) ──
    if nodules:
        z_counts = {}
        for n in nodules:
            z = n['location'][0]
            z_counts[z] = z_counts.get(z, 0) + 1
        best_z = max(z_counts, key=z_counts.get)
    else:
        best_z = ct_scan.shape[0] // 2

    # ── Row 0: Three anatomical views ──
    ax_axial = fig.add_subplot(gs[0, 0:2])
    ax_axial.set_facecolor('#0a0a0a')
    ax_axial.imshow(ct_scan[best_z], cmap='gray', vmin=0, vmax=1)
    ax_axial.set_title(f'Axial View  (Slice {best_z})',
                       fontsize=15, fontweight='bold', color='white', pad=10)
    ax_axial.axis('off')

    ax_coronal = fig.add_subplot(gs[0, 2])
    ax_coronal.set_facecolor('#0a0a0a')
    cor_idx = ct_scan.shape[1] // 2
    ax_coronal.imshow(ct_scan[:, cor_idx, :], cmap='gray', vmin=0, vmax=1,
                      aspect='auto')
    ax_coronal.set_title('Coronal View', fontsize=13, fontweight='bold',
                         color='white', pad=10)
    ax_coronal.axis('off')

    ax_sagittal = fig.add_subplot(gs[0, 3])
    ax_sagittal.set_facecolor('#0a0a0a')
    sag_idx = ct_scan.shape[2] // 2
    ax_sagittal.imshow(ct_scan[:, :, sag_idx], cmap='gray', vmin=0, vmax=1,
                       aspect='auto')
    ax_sagittal.set_title('Sagittal View', fontsize=13, fontweight='bold',
                          color='white', pad=10)
    ax_sagittal.axis('off')

    # ── Draw nodules on all three views ──
    for i, nodule in enumerate(nodules):
        z, y, x = nodule['location']
        radius = max(nodule.get('radius', 10), 8) * 1.5
        _, color, ls = _nodule_risk_style(nodule)
        mal = nodule.get('malignancy_probability', 0.5)

        # Axial view — show if within ±5 slices of displayed slice
        if abs(z - best_z) <= 5:
            ax_axial.add_patch(mpatches.Circle(
                (x, y), radius, lw=3, edgecolor=color,
                facecolor='none', linestyle=ls))
            ax_axial.text(x, y - radius - 8,
                          f"#{i+1}\n{mal*100:.0f}%",
                          color=color, fontsize=11, fontweight='bold',
                          ha='center', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.4',
                                    facecolor='black', alpha=0.8,
                                    edgecolor=color))

        # Coronal view (x, z)
        r_small = radius * 0.6
        ax_coronal.add_patch(mpatches.Circle(
            (x, z), r_small, lw=2, edgecolor=color,
            facecolor='none', linestyle=ls))
        ax_coronal.text(x, z - r_small - 4, f"{i+1}",
                        color=color, fontsize=9, fontweight='bold',
                        ha='center')

        # Sagittal view (y, z)
        ax_sagittal.add_patch(mpatches.Circle(
            (y, z), r_small, lw=2, edgecolor=color,
            facecolor='none', linestyle=ls))
        ax_sagittal.text(y, z - r_small - 4, f"{i+1}",
                         color=color, fontsize=9, fontweight='bold',
                         ha='center')

    # ── Row 1: Summary bar ──
    ax_summary = fig.add_subplot(gs[1, :])
    ax_summary.axis('off')

    if nodules:
        high_n = sum(1 for n in nodules if n.get('malignancy_probability', 0) > 0.70)
        if high_n > 0:
            banner_color, banner_label = RISK_COLORS['high_risk'], 'HIGH RISK'
        elif any(n.get('malignancy_probability', 0) > 0.40 for n in nodules):
            banner_color, banner_label = RISK_COLORS['medium_risk'], 'MEDIUM RISK'
        else:
            banner_color, banner_label = RISK_COLORS['low_risk'], 'LOW RISK'
    else:
        banner_color, banner_label = RISK_COLORS['low_risk'], 'NO NODULES'

    summary = (
        f"Patient: {patient_id}    |    "
        f"Nodules: {len(nodules)}    |    "
        f"Assessment: {banner_label}\n"
        "Legend:  \U0001f534 High Risk (>70%)   "
        "\U0001f7e0 Medium (40-70%)   "
        "\U0001f7e2 Low (<40%)   "
        "\u26aa Uncertain (<60% conf)"
    )
    ax_summary.text(
        0.5, 0.5, summary, fontsize=13, ha='center', va='center',
        color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor=banner_color, alpha=0.85))

    # ── Row 2: Per-nodule details ──
    ax_details = fig.add_subplot(gs[2, :])
    ax_details.axis('off')

    if nodules:
        lines = ['DETAILED NODULE ANALYSIS\n']
        for i, n in enumerate(nodules):
            z, y, x = n['location']
            conf = n.get('detection_confidence', 0) * 100
            mal = n.get('malignancy_probability', 0) * 100
            diam = n.get('radius', 5) * 2
            risk_tag, _, _ = _nodule_risk_style(n)

            if diam < 4:
                rec = 'Routine follow-up (likely benign)'
            elif diam < 8:
                rec = 'Repeat scan in 6-12 months'
            elif diam < 20:
                rec = 'Biopsy or close monitoring recommended'
            else:
                rec = 'Urgent evaluation required'

            lines.append(
                f"Nodule #{i+1} ({risk_tag.replace('_',' ').upper()})\n"
                f"  Location: slice {z}, pos ({x}, {y})    "
                f"Size: ~{diam:.0f}mm    "
                f"Confidence: {conf:.0f}%    "
                f"Malignancy: {mal:.0f}%\n"
                f"  Recommendation: {rec}\n"
            )

        ax_details.text(0.03, 0.95, '\n'.join(lines),
                        fontsize=11, ha='left', va='top',
                        color='white', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.8',
                                  facecolor='#1a1a1a', alpha=0.9))
    else:
        ax_details.text(0.5, 0.5,
                        '✓ No suspicious nodules detected in this scan.',
                        fontsize=14, ha='center', va='center',
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=1',
                                  facecolor=RISK_COLORS['low_risk'], alpha=0.8))

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150,
                facecolor='#0a0a0a', edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.read()).decode('utf-8')


# ── Routes ──

@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'oncovision-x', 'demo_mode': _demo_only})


@app.route('/api/analyze', methods=['POST'])
def analyze_scan():
    """Upload and analyze a CT scan.

    Accepts multipart form with:
      - ct_scan: .mhd / .nii.gz / .nii / .npy / .npz file
      - patient_id: optional
      - scan_date: optional
    """
    try:
        system = get_system()
        if system is None:
            return jsonify(_demo_response(
                request.form.get('patient_id', 'DEMO'),
                request.form.get('scan_date', '')
            ))

        if 'ct_scan' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

        file = request.files['ct_scan']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

        patient_id = request.form.get('patient_id', 'UNKNOWN')
        scan_date = request.form.get('scan_date', '')

        # Save to temp file
        suffix = Path(file.filename).suffix
        if file.filename.endswith('.nii.gz'):
            suffix = '.nii.gz'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            report = system.analyze_patient(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Generate visualization
        viz_b64 = None
        if report.get('ct_scan') is not None and report['num_nodules'] > 0:
            viz_b64 = create_visualization(report['ct_scan'], report['nodules'], patient_id)

        # Build response (strip large arrays)
        nodules_json = []
        for i, n in enumerate(report.get('nodules', [])):
            nodules_json.append({
                'nodule_id': i + 1,
                'location': format_loc(n.get('location')),
                'detection_confidence': round(n.get('detection_confidence', 0) * 100, 1),
                'malignancy_probability': round(n.get('malignancy_probability', 0) * 100, 1),
                'risk_level': n.get('risk_level', 'LOW'),
                'recommendation': n.get('recommendation', ''),
            })

        return jsonify({
            'status': report['status'],
            'patient_id': patient_id,
            'scan_date': scan_date,
            'analysis': {
                'num_nodules_detected': report['num_nodules'],
                'overall_risk': report['patient_risk'],
                'risk_score': round(report.get('patient_risk_score', 0) * 100, 2),
                'nodules': nodules_json,
            },
            'visualization': viz_b64,
            'next_steps': report.get('next_steps', ''),
            'timing': report.get('timing', {}),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/demo-analyze', methods=['POST'])
def demo_analyze():
    """Demo endpoint — returns simulated results without any model."""
    patient_id = request.form.get('patient_id', 'DEMO-001')
    scan_date = request.form.get('scan_date', '2026-03-15')
    return jsonify(_demo_response(patient_id, scan_date))


# Keep old /demo-predict for backward compat
@app.route('/demo-predict', methods=['POST'])
def demo_predict_compat():
    return demo_analyze()


def format_loc(loc):
    if loc is None:
        return 'N/A'
    return f"({loc[0]}, {loc[1]}, {loc[2]})"


def _demo_response(patient_id, scan_date):
    """Generate simulated demo results."""
    from src.pipeline.end_to_end import classify_risk, get_recommendation, get_clinical_recommendation
    demo_nodules = [
        {'id': 1, 'prob': 0.15, 'det': 92.3, 'loc': '(145, 220, 185)'},
        {'id': 2, 'prob': 0.52, 'det': 88.7, 'loc': '(98, 310, 142)'},
        {'id': 3, 'prob': 0.87, 'det': 96.1, 'loc': '(178, 165, 208)'},
    ]

    results = []
    for n in demo_nodules:
        risk = classify_risk(n['prob'])
        results.append({
            'nodule_id': n['id'],
            'location': n['loc'],
            'detection_confidence': n['det'],
            'malignancy_probability': round(n['prob'] * 100, 1),
            'risk_level': risk,
            'recommendation': get_recommendation(risk),
        })

    max_prob = max(n['prob'] for n in demo_nodules)
    overall = classify_risk(max_prob)

    return {
        'status': 'SUCCESS',
        'patient_id': patient_id,
        'scan_date': scan_date,
        'analysis': {
            'num_nodules_detected': len(demo_nodules),
            'overall_risk': overall,
            'risk_score': round(max_prob * 100, 2),
            'nodules': results,
        },
        'visualization': None,
        'next_steps': get_clinical_recommendation(overall),
        'timing': {'preprocess_sec': 1.2, 'detect_sec': 0.8, 'total_sec': 2.3},
    }


# ── Entry ──

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OncoVision-X Web Demo')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--demo', action='store_true',
                        help='Demo-only mode (no model loading)')
    args = parser.parse_args()

    _demo_only = args.demo

    print(f"\n  🫁 OncoVision-X Web Demo")
    print(f"  {'DEMO MODE — no models loaded' if _demo_only else 'Full mode'}")
    print(f"  http://localhost:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
