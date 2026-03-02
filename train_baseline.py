#!/usr/bin/env python3
"""
OncoVision-X Baseline Training Router
================================
Routes baseline training commands to the Universal tracking scripts.
"""
import argparse
import sys
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Router')
    parser.add_argument('--model', type=str, required=True, choices=['resnet3d18', 'resnet2d18'])
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()
    
    cmd = [sys.executable, "scripts/train_experiment.py"]
    cmd.extend(["--experiment", f"baseline_{args.model}"])
        
    if args.config:
        cmd.extend(["--config", args.config])
    if args.resume:
        cmd.append("--resume")
    if args.evaluate:
        cmd.append("--evaluate_only")
    if getattr(args, 'dry_run', False):
        cmd.append("--dry-run")
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
        
    cmd.extend(unknown)
    
    print(f"Routing to Universal Script: {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__))).returncode)

if __name__ == '__main__':
    main()
