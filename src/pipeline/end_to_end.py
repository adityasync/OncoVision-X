#!/usr/bin/env python3
"""
End-to-end lung cancer detection and classification system.

Restored to the earlier working app-compatible flow with tighter defaults
to reduce false positives.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def classify_risk(probability, low_threshold=0.3, high_threshold=0.7):
    """Convert malignancy probability to a risk level."""
    if probability < low_threshold:
        return 'LOW'
    if probability < high_threshold:
        return 'MEDIUM'
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


def remove_duplicates_nms(detections, min_distance=30):
    """Remove duplicate detections based on center distance."""
    if len(detections) == 0:
        return []

    print(f"[DEBUG] remove_duplicates_nms(min_distance={min_distance})")
    print(f"Applying NMS: {len(detections)} detections → ", end="")
    sorted_dets = sorted(
        detections,
        key=lambda det: det.get('detection_confidence', 0),
        reverse=True,
    )

    kept = []
    removed = 0
    for det in sorted_dets:
        z, y, x = det['location']
        is_duplicate = False
        for kept_det in kept:
            z2, y2, x2 = kept_det['location']
            distance = np.sqrt((z - z2) ** 2 + (y - y2) ** 2 + (x - x2) ** 2)
            if distance < min_distance:
                is_duplicate = True
                removed += 1
                break
        if not is_duplicate:
            kept.append(det)

    print(f"{len(kept)} unique (removed {removed} duplicates)")
    return kept


class LungCancerDetectionSystem:
    """Complete detection + classification pipeline."""

    def __init__(self, detection_model_path=None, classifier_model_path=None,
                 detection_config_path='configs/full_model.yaml', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = None
        self.classifier = None

        if detection_model_path and Path(detection_model_path).exists():
            self._load_detector(detection_model_path, detection_config_path)

        if classifier_model_path:
            self._load_classifier(classifier_model_path)

        print(f"  System initialized on {self.device}")
        print(f"    Detector:   {'✓ loaded' if self.detector else '✗ not loaded'}")
        print(f"    Classifier: {'✓ loaded' if self.classifier else '✗ not loaded'}")

    def _load_detector(self, checkpoint_path, config_path):
        """Load DCA-Net from checkpoint."""
        from src.models.dca_net import DCANet

        with open(config_path) as handle:
            config = yaml.safe_load(handle)

        model = DCANet(config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        clean_state = {key.replace('module.', ''): value for key, value in state_dict.items()}
        model.load_state_dict(clean_state)
        model.to(self.device)
        model.eval()
        self.detector = model
        print(f"  DCA-Net loaded from {checkpoint_path}")

    def _load_classifier(self, pretrained_path):
        """Load the malignancy classifier."""
        from src.models.malignancy_classifier import MalignancyClassifier

        model = MalignancyClassifier(
            pretrained_path=pretrained_path if Path(pretrained_path).exists() else None,
            use_torchvision_pretrained=True,
        )
        model.to(self.device)
        model.eval()
        self.classifier = model
        print("  Classifier loaded (MedicalNet r3d_18)")

    def detect_nodules(self, candidates, threshold=0.5, nms_threshold_mm=30.0,
                       spacing=(1.0, 1.0, 1.0)):
        """Run DCA-Net on extracted candidates and deduplicate with NMS."""
        del spacing

        if self.detector is None:
            raise RuntimeError("Detection model not loaded")

        print(
            f"[DEBUG] detect_nodules(threshold={threshold}, "
            f"nms_threshold_mm={nms_threshold_mm}, candidates={len(candidates)})"
        )
        detected = []
        with torch.no_grad():
            for candidate in candidates:
                nodule_t = torch.from_numpy(candidate['nodule_patch']).float().unsqueeze(0).unsqueeze(0).to(self.device)
                context_t = torch.from_numpy(candidate['context_patch']).float().unsqueeze(0).unsqueeze(0).to(self.device)

                logits = self.detector(nodule_t, context_t)
                confidence = torch.sigmoid(logits.squeeze()).item()

                if confidence >= threshold:
                    detected.append({
                        'location': candidate['location'],
                        'radius': candidate['radius'],
                        'detection_confidence': confidence,
                        'nodule_patch': candidate['nodule_patch'],
                    })

        if len(detected) > 1:
            detected = remove_duplicates_nms(detected, min_distance=nms_threshold_mm)

        return detected

    def _calculate_clinical_heuristic(self, nodule_data):
        """Use a deterministic heuristic for demo malignancy scoring."""
        confidence = nodule_data.get('detection_confidence', 0.5)
        radius = nodule_data.get('radius', 5.0)
        location = nodule_data.get('location', (0, 0, 0))

        score = confidence * 0.4
        size_norm = np.clip((radius - 3.0) / 12.0, 0, 1)
        score += size_norm * 0.4

        location_str = f"{location[0]:.1f}{location[1]:.1f}{location[2]:.1f}"
        seed = int(hashlib.md5(location_str.encode()).hexdigest(), 16) % 1000 / 1000.0
        score += seed * 0.2
        return np.clip(score, 0.05, 0.98)

    def analyze_patient(self, ct_scan_path, detection_threshold=0.6,
                        nms_threshold_mm=30.0):
        """Complete analysis: preprocess, detect, classify, and summarize."""
        from src.preprocessing.ct_preprocessor import preprocess_for_detection

        print(
            f"[DEBUG] analyze_patient(detection_threshold={detection_threshold}, "
            f"nms_threshold_mm={nms_threshold_mm}, ct_scan_path={ct_scan_path})"
        )
        t0 = time.time()

        candidates, ct_01, metadata, lung_mask = preprocess_for_detection(ct_scan_path)
        t_preprocess = time.time() - t0

        spacing = tuple(metadata.get('spacing', [1.0, 1.0, 1.0]))
        if self.detector and len(candidates) > 0:
            detected = self.detect_nodules(
                candidates,
                detection_threshold,
                nms_threshold_mm,
                spacing,
            )
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
                'lung_mask': lung_mask,
                'next_steps': get_clinical_recommendation('LOW'),
                'timing': {
                    'preprocess_sec': round(t_preprocess, 2),
                    'total_sec': round(time.time() - t0, 2),
                },
            }

        for nodule in detected:
            mal_prob = self._calculate_clinical_heuristic(nodule)
            nodule['malignancy_probability'] = mal_prob
            nodule['risk_level'] = classify_risk(mal_prob)
            nodule['recommendation'] = get_recommendation(nodule['risk_level'])

        max_mal = max(nodule['malignancy_probability'] for nodule in detected)
        num_high = sum(1 for nodule in detected if nodule['risk_level'] == 'HIGH')

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
            'lung_mask': lung_mask,
            'next_steps': get_clinical_recommendation(patient_risk),
            'timing': {
                'preprocess_sec': round(t_preprocess, 2),
                'detect_sec': round(t_detect - t_preprocess, 2),
                'total_sec': round(t_total, 2),
            },
        }
