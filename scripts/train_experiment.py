#!/usr/bin/env python3
"""
Universal Training Script for All Experiments
"""

import argparse
import sys
import os
import yaml
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.experiment_manager import ExperimentManager
from src.utils.logging_utils import setup_logging
from src.data.dataset import create_data_loaders
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train experiment')
    
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=[
            'full_model',
            'ablation_no_context',
            'ablation_no_attention',
            'ablation_no_curriculum',
            'ablation_no_uncertainty',
            'baseline_resnet3d18',
            'baseline_resnet2d18'
        ],
        help='Experiment name'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional, will use default)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    parser.add_argument(
        '--evaluate_only',
        action='store_true',
        help='Only run evaluation on test set'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run only a few batches for debugging'
    )
    
    return parser.parse_args()


def load_config(experiment_name, config_path=None):
    """Load experiment configuration"""
    if config_path is None:
        config_path = f'configs/{experiment_name}.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_model(config):
    """Create model based on config"""
    model_type = config.get('model', {}).get('type', 'dca_net')
    
    if model_type == 'dca_net':
        ablation = config.get('model', {}).get('ablation', None)
        if not ablation and 'ablation' in config:
            ablation = config['ablation']
        
        from src.models.dca_net import DCANet
        model = DCANet(config) # The constructor handles the ablation internally 
    
    elif model_type == 'resnet3d18':
        from src.models.baselines import ResNet3D18
        num_classes = config.get('model', {}).get('num_classes', 1)
        model = ResNet3D18(num_classes=num_classes)
    
    elif model_type == 'resnet2d18':
        from src.models.baselines import ResNet2D18SliceLevel
        num_classes = config.get('model', {}).get('num_classes', 1)
        model = ResNet2D18SliceLevel(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_experiment(args):
    """Main training function"""
    
    # 1. Setup experiment manager
    exp_manager = ExperimentManager(args.experiment)
    
    # 2. Load configuration
    config = load_config(args.experiment, args.config)
    
    if args.batch_size:
        if 'training' not in config: config['training'] = {}
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        if 'training' not in config: config['training'] = {}
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        if 'training' not in config: config['training'] = {}
        config['training']['learning_rate'] = args.lr
        
    # Override logging/checkpoint paths in config to use experiment manager
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['log_dir'] = str(exp_manager.dirs['logs'])
    config['logging']['checkpoint_dir'] = str(exp_manager.dirs['checkpoints'])
    
    if 'paths' not in config:
        config['paths'] = {}
    config['paths']['log_dir'] = str(exp_manager.dirs['logs'])
    config['paths']['model_save_dir'] = str(exp_manager.dirs['checkpoints'])
    
    exp_manager.save_config(config)
    
    # 3. Setup logging
    logger = setup_logging(config)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {args.experiment}")
    print(f"{'='*60}\n")
    
    # 4. Create model
    print("Creating model...")
    model = create_model(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # 5. Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # 6. Evaluation Only Mode
    if args.evaluate_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        best_checkpoint_path = exp_manager.get_checkpoint_path('best')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {best_checkpoint_path}")
        else:
            print("No checkpoint found to evaluate!")
            return
            
        evaluator = Evaluator(model, test_loader, device=device, logger=logger)
        results = evaluator.evaluate(output_dir=str(exp_manager.dirs['metrics']), run_uncertainty=True)
        return
        
    # 7. Setup Training
    print("\nSetting up training...")
    trainer = Trainer(model, config, train_loader, val_loader, logger=logger)
    
    if args.resume:
        last_checkpoint_path = exp_manager.get_checkpoint_path('last')
        if os.path.exists(last_checkpoint_path):
            trainer.load_checkpoint(last_checkpoint_path)
            print(f"Resumed from {last_checkpoint_path}")
            
    # 8. Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}\n")
    
    num_epochs = config.get('training', {}).get('num_epochs', 60)
    trainer.train(num_epochs=num_epochs, dry_run=args.dry_run)
    
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    
    # Load best checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_checkpoint_path = exp_manager.get_checkpoint_path('best')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        evaluator = Evaluator(model, test_loader, device=device, logger=logger)
        test_results = evaluator.evaluate(output_dir=str(exp_manager.dirs['metrics']), run_uncertainty=True)
        
        # Save results map
        final_results = {
            'experiment': args.experiment,
            'best_epoch': checkpoint.get('epoch', 0),
            'best_val_auc': checkpoint.get('val_auc', 0.0),
            'test_results': test_results,
            'config': config
        }
        
        exp_manager.save_results(final_results)
        exp_manager.save_metadata()
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {exp_manager.exp_dir}")

if __name__ == '__main__':
    args = parse_args()
    train_experiment(args)
