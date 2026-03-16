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


def classify_size_bucket(radius):
    """Group candidate size into coarse clinical buckets."""
    diameter_mm = max(radius * 2.0, 0.0)
    if diameter_mm < 6:
        return 'SMALL'
    if diameter_mm < 10:
        return 'INTERMEDIATE'
    return 'LARGE'


def describe_nodule_characteristics(nodule_data):
    """Build structured descriptors used by recommendation templates."""
    confidence = float(nodule_data.get('detection_confidence', 0.0))
    malignancy = float(nodule_data.get('malignancy_probability', 0.0))
    radius = float(nodule_data.get('radius', 0.0))
    size_bucket = classify_size_bucket(radius)
    return {
        'confidence': confidence,
        'malignancy': malignancy,
        'radius': radius,
        'size_bucket': size_bucket,
        'high_confidence': confidence >= 0.8,
        'borderline_confidence': confidence < 0.6,
        'marked_malignancy': malignancy >= 0.75,
        'moderate_malignancy': 0.45 <= malignancy < 0.75,
    }


def summarize_distribution(nodules):
    """Describe whether findings are solitary, paired, or multifocal."""
    count = len(nodules)
    if count <= 1:
        return 'solitary'
    if count == 2:
        return 'paired'
    return 'multifocal'


def get_recommendation(nodule_data, total_nodules=1):
    """Per-nodule recommendation with richer condition-based guidance."""
    risk_level = str(nodule_data.get('risk_level', 'LOW')).upper()
    traits = describe_nodule_characteristics(nodule_data)
    location_text = str(nodule_data.get('location', 'N/A'))

    if risk_level == 'HIGH':
        if traits['marked_malignancy'] and traits['size_bucket'] == 'LARGE':
            recommendation = 'Dominant suspicious lesion identified; urgent pulmonology or thoracic review is advised and tissue diagnosis planning should be considered.'
        elif traits['high_confidence']:
            recommendation = 'High-confidence suspicious finding; expedited specialist review is advised with short-interval diagnostic follow-up.'
        else:
            recommendation = 'Prompt physician review is advised to confirm the finding and define the next diagnostic step.'
    elif risk_level == 'MEDIUM':
        if traits['size_bucket'] == 'LARGE':
            recommendation = 'Diagnostic chest CT follow-up within 2-3 months is recommended because of lesion size.'
        elif traits['moderate_malignancy'] and traits['high_confidence']:
            recommendation = 'Short-interval follow-up imaging in about 3 months is recommended.'
        elif traits['borderline_confidence']:
            recommendation = 'Correlate with prior imaging and consider repeat CT in 3-6 months if clinically appropriate.'
        else:
            recommendation = 'Follow-up CT in 3-6 months is recommended to document stability.'
    else:
        if traits['size_bucket'] == 'LARGE':
            recommendation = 'Although risk is low, lesion size supports interval imaging to confirm stability.'
        elif traits['borderline_confidence']:
            recommendation = 'Low-risk finding; compare with prior scans and continue routine surveillance.'
        else:
            recommendation = 'Continue routine screening and document this finding for interval comparison.'

    if traits['size_bucket'] == 'LARGE':
        recommendation += ' The apparent lesion size merits direct comparison with prior studies if available.'
    elif traits['size_bucket'] == 'SMALL' and risk_level != 'HIGH':
        recommendation += ' Small lesion size favors surveillance over immediate invasive workup unless clinical history suggests otherwise.'

    if traits['borderline_confidence']:
        recommendation += ' Confidence is limited, so correlation with multiplanar review and prior imaging is especially important.'
    elif traits['high_confidence']:
        recommendation += ' Detection confidence is high enough to prioritize this focus during review.'

    if total_nodules >= 3 and risk_level != 'HIGH':
        recommendation += ' Because multiple nodules are present, compare all visible foci with prior studies at the next review.'
    elif total_nodules == 2 and risk_level != 'HIGH':
        recommendation += ' A paired-finding pattern is present, so interval comparison of both sites is recommended.'

    recommendation += f' Reported location for review: {location_text}.'

    return recommendation


def get_clinical_recommendation(patient_risk, nodules):
    """Overall patient-level recommendation with aggregate finding context."""
    num_nodules = len(nodules)
    max_malignancy = max((float(nodule.get('malignancy_probability', 0.0)) for nodule in nodules), default=0.0)
    max_confidence = max((float(nodule.get('detection_confidence', 0.0)) for nodule in nodules), default=0.0)
    largest_radius = max((float(nodule.get('radius', 0.0)) for nodule in nodules), default=0.0)
    size_bucket = classify_size_bucket(largest_radius)
    distribution = summarize_distribution(nodules)

    if num_nodules == 0:
        return 'No immediate action is suggested. Continue routine screening and retain this study for future comparison.'

    if patient_risk == 'HIGH':
        if num_nodules >= 2:
            return 'Multiple suspicious findings are present. Prompt pulmonology or thoracic review is recommended, with diagnostic workup prioritized and all dominant foci correlated across views.'
        if size_bucket == 'LARGE' or max_malignancy >= 0.8:
            return 'A dominant suspicious lesion is present. Urgent specialist assessment and diagnostic planning are recommended.'
        return 'High-risk imaging findings warrant prompt physician review and short-interval diagnostic follow-up.'

    if patient_risk == 'MEDIUM':
        if num_nodules >= 3:
            return 'Several indeterminate nodules are present. Short-interval CT follow-up and comparison with prior scans are recommended.'
        if size_bucket == 'LARGE':
            return 'An intermediate-risk lesion with substantial size is present. Repeat diagnostic imaging within 2-3 months is recommended.'
        if max_confidence < 0.6:
            return 'Findings are indeterminate with limited confidence. Correlation with prior imaging and clinical context is recommended.'
        return f'Schedule follow-up CT in 3-6 months and compare this {distribution} finding pattern with any prior chest imaging.'

    if num_nodules >= 3:
        return 'Low-risk multifocal findings are present. Routine surveillance with prior-study comparison is recommended.'
    if size_bucket == 'LARGE':
        return 'Overall risk is low, but lesion size supports interval imaging to confirm stability.'
    if max_confidence < 0.6:
        return 'No immediate intervention is suggested. Keep this study for comparison and continue routine screening.'
    return f'No immediate action is required. Continue routine screening and compare this {distribution} finding pattern with future studies for stability.'


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
                'next_steps': get_clinical_recommendation('LOW', []),
                'timing': {
                    'preprocess_sec': round(t_preprocess, 2),
                    'total_sec': round(time.time() - t0, 2),
                },
            }

        for nodule in detected:
            mal_prob = self._calculate_clinical_heuristic(nodule)
            nodule['malignancy_probability'] = mal_prob
            nodule['risk_level'] = classify_risk(mal_prob)
            nodule['recommendation'] = get_recommendation(nodule, total_nodules=len(detected))

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
            'next_steps': get_clinical_recommendation(patient_risk, detected),
            'timing': {
                'preprocess_sec': round(t_preprocess, 2),
                'detect_sec': round(t_detect - t_preprocess, 2),
                'total_sec': round(t_total, 2),
            },
        }
