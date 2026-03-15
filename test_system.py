#!/usr/bin/env python3
"""Quick test of the complete rebuilt system."""

import os
import sys
from pathlib import Path

from src.pipeline.end_to_end import LungCancerDetectionSystem


DEFAULT_SCAN = (
    'data/subset0/subset0/'
    '1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd'
)


def resolve_path(env_name, default_value):
    """Return a file path only if it exists."""
    path = os.environ.get(env_name, default_value)
    return path if Path(path).exists() else None


def main():
    print("\n" + "=" * 60)
    print("TESTING COMPLETE SYSTEM")
    print("=" * 60 + "\n")

    system = LungCancerDetectionSystem(
        detection_model_path=resolve_path('DETECTION_CHECKPOINT', 'experiments/full_model/checkpoints/best.pth'),
        classifier_model_path=resolve_path('CLASSIFIER_CHECKPOINT', 'pretrained/resnet_18_23dataset.pth'),
    )

    scan = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCAN
    result = system.analyze_patient(scan)

    print("=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Nodules: {result['num_nodules']}")
    print(f"Patient Risk: {result['patient_risk']}")
    print()

    for idx, nodule in enumerate(result['nodules'], start=1):
        print(f"Nodule #{idx}:")
        print(f"  Location: {nodule['location']}")
        print(f"  Detection: {nodule.get('detection_confidence', 0):.1%}")
        print(f"  Malignancy: {nodule.get('malignancy_probability', 0):.1%}")
        print(f"  Risk: {nodule.get('risk_level', 'LOW')}")
        print()

    print("=" * 60)
    print("Test complete")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
