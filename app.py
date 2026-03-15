#!/usr/bin/env python3
"""Flask app for the rebuilt lung cancer detection demo."""

import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


app = Flask(__name__, static_folder='frontend')
CORS(app)

_system = None


def get_system():
    """Lazy-load the detection system."""
    global _system
    if _system is not None:
        return _system

    from src.pipeline.end_to_end import LungCancerDetectionSystem

    detection_ckpt = os.environ.get(
        'DETECTION_CHECKPOINT',
        'experiments/full_model/checkpoints/best.pth',
    )
    classifier_ckpt = os.environ.get(
        'CLASSIFIER_CHECKPOINT',
        'pretrained/resnet_18_23dataset.pth',
    )
    detection_cfg = os.environ.get(
        'DETECTION_CONFIG',
        'configs/full_model.yaml',
    )

    _system = LungCancerDetectionSystem(
        detection_model_path=detection_ckpt if Path(detection_ckpt).exists() else None,
        classifier_model_path=classifier_ckpt if Path(classifier_ckpt).exists() else None,
        detection_config_path=detection_cfg,
    )
    return _system


def create_visualization(ct_scan, nodules):
    """Create a 3-panel CT visualization with nodule overlays."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0a0a0a')

    if nodules:
        z_counts = {}
        for nodule in nodules:
            z_value = nodule['location'][0]
            z_counts[z_value] = z_counts.get(z_value, 0) + 1
        best_z = max(z_counts, key=z_counts.get)
    else:
        best_z = ct_scan.shape[0] // 2

    cor_idx = ct_scan.shape[1] // 2
    sag_idx = ct_scan.shape[2] // 2

    axes[0].imshow(ct_scan[best_z], cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(ct_scan[:, cor_idx, :], cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[2].imshow(ct_scan[:, :, sag_idx], cmap='gray', vmin=0, vmax=1, aspect='auto')

    titles = [f'Axial View (Slice {best_z})', 'Coronal View', 'Sagittal View']
    for axis, title in zip(axes, titles):
        axis.set_title(title, color='white', fontsize=14, fontweight='bold')
        axis.axis('off')
        axis.set_facecolor('#0a0a0a')

    colors = {'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'}
    for idx, nodule in enumerate(nodules):
        z, y, x = nodule['location']
        radius = max(nodule.get('radius', 8), 6) * 1.5
        color = colors.get(nodule.get('risk_level', 'MEDIUM'), '#ffc107')

        if abs(z - best_z) <= 5:
            axes[0].add_patch(mpatches.Circle((x, y), radius, lw=3, edgecolor=color, facecolor='none'))
            axes[0].text(
                x,
                y - radius - 8,
                f"#{idx + 1}",
                color=color,
                fontsize=12,
                fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor=color),
            )

        axes[1].add_patch(mpatches.Circle((x, z), radius * 0.6, lw=2, edgecolor=color, facecolor='none'))
        axes[1].text(
            x,
            z - radius * 0.6 - 4,
            f"#{idx + 1}",
            color=color,
            fontsize=10,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'),
        )

        axes[2].add_patch(mpatches.Circle((y, z), radius * 0.6, lw=2, edgecolor=color, facecolor='none'))
        axes[2].text(
            y,
            z - radius * 0.6 - 4,
            f"#{idx + 1}",
            color=color,
            fontsize=10,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'),
        )

    plt.tight_layout(pad=2.0)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route('/')
def index():
    """Serve the UI."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/health')
def health():
    """Basic health check."""
    return jsonify({'status': 'healthy'})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint."""
    try:
        system = get_system()
        files = request.files.getlist('ct_scan')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            primary = None
            for upload in files:
                target = os.path.join(tmpdir, upload.filename)
                upload.save(target)
                ext = Path(upload.filename).suffix.lower()
                if upload.filename.endswith('.nii.gz'):
                    ext = '.nii.gz'
                if ext in ['.mhd', '.nii', '.nii.gz', '.npz', '.npy']:
                    primary = target

            if not primary:
                return jsonify({'error': 'No valid scan file'}), 400

            report = system.analyze_patient(primary)
            visualization = None
            if report.get('ct_scan') is not None and report['num_nodules'] > 0:
                visualization = create_visualization(report['ct_scan'], report['nodules'])

        nodules_json = []
        for idx, nodule in enumerate(report.get('nodules', []), start=1):
            nodules_json.append({
                'nodule_id': idx,
                'location': f"({nodule['location'][0]}, {nodule['location'][1]}, {nodule['location'][2]})",
                'detection_confidence': round(nodule.get('detection_confidence', 0) * 100, 1),
                'malignancy_probability': round(nodule.get('malignancy_probability', 0) * 100, 1),
                'risk_level': nodule.get('risk_level', 'LOW'),
                'recommendation': nodule.get('recommendation', 'Consult physician'),
            })

        return jsonify({
            'status': report['status'],
            'next_steps': report.get('next_steps', 'Consult physician for evaluation.'),
            'analysis': {
                'num_nodules_detected': report['num_nodules'],
                'overall_risk': report['patient_risk'],
                'risk_score': round(report.get('patient_risk_score', 0) * 100, 1),
                'nodules': nodules_json,
            },
            'visualization': visualization,
            'timing': report.get('timing', {}),
        })

    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/<path:path>')
def frontend_files(path):
    """Serve frontend assets from the frontend directory."""
    return send_from_directory(app.static_folder, path)


if __name__ == '__main__':
    print("\nOncoVision-X Web Demo\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
