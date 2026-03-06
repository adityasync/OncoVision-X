#!/usr/bin/env python3
"""
Verify that the context stream is actually used and contributes to predictions.

Usage:
    python scripts/verify_context_stream.py --checkpoint experiments/full_model/checkpoints/best.pth
"""

import argparse
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dca_net import DCANet


def verify_context_stream(checkpoint_path):
    """Verify context stream contributes to model predictions."""
    
    print("=" * 70)
    print("CONTEXT STREAM VERIFICATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Create model
    model = DCANet(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Check model has context stream
    if model.context_stream is None:
        print("\n❌ FAILED: Model has no context stream (ablation=no_context)")
        return False
    
    print(f"\n✅ Context stream exists")
    print(f"   Ablation mode: {model.ablation}")
    
    # Run multiple random tests
    num_tests = 20
    differences = []
    
    print(f"\nRunning {num_tests} random tests...")
    
    with torch.no_grad():
        for i in range(num_tests):
            # Create random nodule patch
            nodule = torch.randn(1, 1, 64, 64, 64).to(device)
            
            # Real random context vs zero context
            context_real = torch.randn(1, 1, 48, 48, 48).to(device)
            context_zeros = torch.zeros(1, 1, 48, 48, 48).to(device)
            
            # Get predictions
            logits_real = model(nodule, context_real)
            logits_zeros = model(nodule, context_zeros)
            
            prob_real = torch.sigmoid(logits_real).item()
            prob_zeros = torch.sigmoid(logits_zeros).item()
            
            diff = abs(prob_real - prob_zeros)
            differences.append(diff)
            
            if i < 5:  # Show first 5
                print(f"  Test {i+1}: real={prob_real:.6f} zeros={prob_zeros:.6f} diff={diff:.6f}")
    
    mean_diff = np.mean(differences)
    max_diff = np.max(differences)
    min_diff = np.min(differences)
    
    print(f"\n{'─' * 50}")
    print(f"  Mean prediction difference: {mean_diff:.6f}")
    print(f"  Min difference:             {min_diff:.6f}")
    print(f"  Max difference:             {max_diff:.6f}")
    print(f"{'─' * 50}")
    
    if mean_diff < 0.001:
        print(f"\n❌ FAILED: Context stream is NOT contributing to predictions!")
        print(f"   Mean difference {mean_diff:.6f} < threshold 0.001")
        return False
    elif mean_diff < 0.01:
        print(f"\n⚠️  WARNING: Context stream has minimal influence")
        print(f"   Mean difference {mean_diff:.6f} < recommended 0.01")
        return True  # Technically passing but weak
    else:
        print(f"\n✅ PASSED: Context stream IS contributing to predictions")
        print(f"   Mean difference {mean_diff:.6f} > threshold 0.01")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify context stream')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    success = verify_context_stream(args.checkpoint)
    sys.exit(0 if success else 1)
