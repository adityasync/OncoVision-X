#!/usr/bin/env python3
"""
DCA-Net Comprehensive Evaluation Router
================================
Routes evaluation commands to the Universal tracking scripts.
"""
import argparse
import sys
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Router")
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--model', type=str, default=None, choices=['resnet3d18', 'resnet2d18'])
    parser.add_argument('--ablation', type=str, default=None)
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()
    
    cmd = [sys.executable, "scripts/evaluate_experiment.py"]
    
    if args.model:
        cmd.extend(["--experiment", f"baseline_{args.model}"])
    elif args.ablation:
        cmd.extend(["--experiment", f"ablation_{args.ablation}"])
    else:
        cmd.extend(["--experiment", "full_model"])
        
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
        
    # Forward unknown args (like split, etc.)
    cmd.extend(unknown)
    
    print(f"Routing to Universal Evaluation Script: {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__))).returncode)

if __name__ == '__main__':
    main()
