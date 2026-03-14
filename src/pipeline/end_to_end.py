#!/usr/bin/env python3
"""
End-to-End Lung Cancer Detection & Classification System

Complete pipeline:
  CT Scan → Preprocess → DCA-Net Detection → Malignancy Classification → Risk Report

Optimized for RTX 3050 (4GB VRAM) or CPU inference.
Uses our trained DCA-Net for detection + pretrained MedicalNet for classification.

NOT part of the research paper — demo feature only.
"""

import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path


# ── Risk helpers ──

def classify_risk(probability, low_threshold=0.3, high_threshold=0.7):
    """Convert malignancy probability → risk level."""
    if probability < low_threshold:
        return 'LOW'
    elif probability < high_threshold:
        return 'MEDIUM'
    else:
        return 'HIGH'


def get_recommendation(risk_level):
    """Per-nodule clinical recommendation."""
    return {
        'LOW': 'Continue routine screening',
        'MEDIUM': 'Follow-up scan in 6 months recommended',
        'HIGH': 'Immediate clinical evaluation recommended',
    }.get(risk_level, 'Consult physician')


def get_clinical_recommendation(patient_risk):
    """Overall patient-level recommendation."""
    return {
        'LOW': 'No immediate action required. Continue routine screening.',
        'MEDIUM': 'Schedule follow-up CT in 3-6 months. Monitor for changes.',
        'HIGH': 'HIGH RISK: Immediate referral to pulmonologist recommended. Consider biopsy.',
    }.get(patient_risk, 'Consult physician for evaluation.')


def format_location(location):
    """Format (z, y, x) voxel coords for display."""
    if location is None:
        return 'N/A'
    return f"({location[0]}, {location[1]}, {location[2]})"


def deduplicate_nodules(detections, spacing=(1.0, 1.0, 1.0), threshold_mm=20.0):
    """Non-Maximum Suppression: merge detections of the same nodule.

    Multiple blob candidates on adjacent CT slices often detect the same
    physical nodule. This merges detections within `threshold_mm` of each
    other, keeping the one with highest detection confidence.

    Args:
        detections: List of dicts with 'location' and 'detection_confidence'
        spacing: CT voxel spacing (z, y, x) in mm
        threshold_mm: Max 3D distance to consider as same nodule (default 20mm)

    Returns:
        Deduplicated list (subset of input, highest-confidence per cluster)
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending — keep best detection per cluster
    detections = sorted(detections,
                        key=lambda d: d['detection_confidence'],
                        reverse=True)

    spacing = np.array(spacing, dtype=np.float64)
    unique = []
    suppressed = set()

    for i, det_i in enumerate(detections):
        if i in suppressed:
            continue
        unique.append(det_i)
        loc_i = np.array(det_i['location'], dtype=np.float64)

        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            loc_j = np.array(detections[j]['location'], dtype=np.float64)
            distance_mm = np.linalg.norm((loc_i - loc_j) * spacing)
            if distance_mm < threshold_mm:
                suppressed.add(j)

    print(f"  NMS: {len(detections)} detections → {len(unique)} unique nodules "
          f"(threshold={threshold_mm}mm)")
    return unique


# ── Main System ──

class LungCancerDetectionSystem:
    """Complete detection + classification pipeline.

    Args:
        detection_model_path: Path to trained DCA-Net checkpoint
        classifier_model_path: Path to MedicalNet/pretrained classifier weights
        detection_config_path: Path to DCA-Net config YAML
        device: 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(self, detection_model_path=None, classifier_model_path=None,
                 detection_config_path='configs/full_model.yaml', device=None):

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.detector = None
        self.classifier = None

        # Load detection model (DCA-Net)
        if detection_model_path and Path(detection_model_path).exists():
            self._load_detector(detection_model_path, detection_config_path)

        # Load classification model (MedicalNet / pretrained)
        if classifier_model_path:
            self._load_classifier(classifier_model_path)

        print(f"  System initialized on {self.device}")
        print(f"    Detector:   {'✓ loaded' if self.detector else '✗ not loaded'}")
        print(f"    Classifier: {'✓ loaded' if self.classifier else '✗ not loaded'}")

    def _load_detector(self, checkpoint_path, config_path):
        """Load DCA-Net from checkpoint."""
        from src.models.dca_net import DCANet

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model = DCANet(config)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state)
        model.to(self.device)
        model.eval()
        self.detector = model
        print(f"  DCA-Net loaded from {checkpoint_path}")

    def _load_classifier(self, pretrained_path):
        """Load MedicalNet malignancy classifier."""
        from src.models.malignancy_classifier import MalignancyClassifier

        model = MalignancyClassifier(
            pretrained_path=pretrained_path if Path(pretrained_path).exists() else None,
            use_torchvision_pretrained=True,
        )
        model.to(self.device)
        model.eval()
        self.classifier = model
        print(f"  Classifier loaded (MedicalNet r3d_18)")

    # ── Detection ──

    def detect_nodules(self, candidates, threshold=0.5, nms_threshold_mm=15.0,
                       spacing=(1.0, 1.0, 1.0)):
        """Run DCA-Net on extracted candidates, then deduplicate with NMS.

        Args:
            candidates: List from ct_preprocessor.preprocess_for_detection()
            threshold: Detection confidence threshold
            nms_threshold_mm: 3D distance threshold for deduplication (mm)
            spacing: Voxel spacing (z, y, x) in mm

        Returns:
            List of unique detected nodules
        """
        if self.detector is None:
            raise RuntimeError("Detection model not loaded")

        detected = []
        with torch.no_grad():
            for cand in candidates:
                nodule_t = torch.from_numpy(cand['nodule_patch']).float()
                context_t = torch.from_numpy(cand['context_patch']).float()

                nodule_t = nodule_t.unsqueeze(0).unsqueeze(0).to(self.device)
                context_t = context_t.unsqueeze(0).unsqueeze(0).to(self.device)

                logits = self.detector(nodule_t, context_t)
                confidence = torch.sigmoid(logits.squeeze()).item()

                if confidence >= threshold:
                    detected.append({
                        'location': cand['location'],
                        'radius': cand['radius'],
                        'detection_confidence': confidence,
                        'nodule_patch': cand['nodule_patch'],
                    })

        # Apply NMS to merge duplicate detections of the same nodule
        if len(detected) > 1:
            detected = deduplicate_nodules(detected, spacing, nms_threshold_mm)

        return detected

    # ── Classification ──

    def classify_nodule(self, nodule_patch_64):
        """Classify a single nodule patch for malignancy.

        Args:
            nodule_patch_64: numpy (64, 64, 64) nodule patch

        Returns:
            malignancy_prob: float [0, 1]
        """
        if self.classifier is None:
            raise RuntimeError("Classification model not loaded")

        from src.preprocessing.ct_preprocessor import extract_classification_patch

        # Crop center 32³ from 64³ detection patch
        patch_32 = extract_classification_patch(nodule_patch_64, None, size=32)

        patch_t = torch.from_numpy(patch_32).float()
        patch_t = patch_t.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 32, 32, 32)

        with torch.no_grad():
            prob = self.classifier(patch_t)
            return prob.squeeze().item()

    # ── Full Pipeline ──

    def analyze_patient(self, ct_scan_path, detection_threshold=0.5,
                        nms_threshold_mm=20.0):
        """Complete analysis: detect nodules → classify each → patient risk.

        Args:
            ct_scan_path: Path to .mhd / .nii.gz file
            detection_threshold: Min confidence to keep a detection
            nms_threshold_mm: 3D distance threshold for NMS deduplication

        Returns:
            dict with status, nodules, patient_risk, ct_scan (for viz), etc.
        """
        from src.preprocessing.ct_preprocessor import preprocess_for_detection

        t0 = time.time()

        # Step 1: Preprocess & find candidates
        candidates, ct_01, metadata = preprocess_for_detection(ct_scan_path)
        t_preprocess = time.time() - t0

        # Step 2: Detect nodules (with NMS deduplication)
        spacing = tuple(metadata.get('spacing', [1.0, 1.0, 1.0]))
        if self.detector and len(candidates) > 0:
            detected = self.detect_nodules(
                candidates, detection_threshold, nms_threshold_mm, spacing)
        else:
            detected = []
        t_detect = time.time() - t0

        if len(detected) == 0:
            return {
                'status': 'NO_NODULES_DETECTED',
                'num_nodules': 0,
                'patient_risk': 'LOW',
                'patient_risk_score': 0.0,
                'message': 'No suspicious nodules detected. Scan appears clear.',
                'nodules': [],
                'ct_scan': ct_01,
                'metadata': metadata,
                'next_steps': get_clinical_recommendation('LOW'),
                'timing': {
                    'preprocess_sec': round(t_preprocess, 2),
                    'total_sec': round(time.time() - t0, 2),
                },
            }

        # Step 3: Classify each detected nodule
        for nodule in detected:
            if self.classifier:
                mal_prob = self.classify_nodule(nodule['nodule_patch'])
            else:
                # Fallback: use detection confidence as rough proxy
                mal_prob = nodule['detection_confidence'] * 0.6
            nodule['malignancy_probability'] = mal_prob
            nodule['risk_level'] = classify_risk(mal_prob)
            nodule['recommendation'] = get_recommendation(nodule['risk_level'])

        # Step 4: Patient-level risk
        max_mal = max(n['malignancy_probability'] for n in detected)
        num_high = sum(1 for n in detected if n['risk_level'] == 'HIGH')

        if num_high > 0:
            patient_risk = 'HIGH'
        elif max_mal > 0.5:
            patient_risk = 'MEDIUM'
        else:
            patient_risk = 'LOW'

        t_total = time.time() - t0

        return {
            'status': 'SUCCESS',
            'num_nodules': len(detected),
            'nodules': detected,
            'patient_risk': patient_risk,
            'patient_risk_score': max_mal,
            'ct_scan': ct_01,
            'metadata': metadata,
            'next_steps': get_clinical_recommendation(patient_risk),
            'timing': {
                'preprocess_sec': round(t_preprocess, 2),
                'detect_sec': round(t_detect - t_preprocess, 2),
                'total_sec': round(t_total, 2),
            },
        }
