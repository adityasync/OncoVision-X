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
import numpy as np
import random
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.experiment_manager import ExperimentManager
from src.utils.logging_utils import setup_logging
from src.data.dataset import create_data_loaders
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(experiment):
    print(f"""
{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║    ██████╗ ███╗   ██╗ ██████╗ ██████╗                              ║
║   ██╔═══██╗████╗  ██║██╔════╝██╔═══██╗                             ║
║   ██║   ██║██╔██╗ ██║██║     ██║   ██║                             ║
║   ██║   ██║██║╚██╗██║██║     ██║   ██║                             ║
║   ╚██████╔╝██║ ╚████║╚██████╗╚██████╔╝                             ║
║    ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝                              ║
║   ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ██╗  ██╗           ║
║   ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ╚██╗██╔╝           ║
║   ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║     ╚███╔╝            ║
║   ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║     ██╔██╗            ║
║    ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ██╔╝ ██╗           ║
║     ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝  ╚═╝           ║
║                                                                    ║
║   Dual-Context Attention Network                                   ║
║   Experiment: {experiment:<49s}║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝{RESET}
""")


def section(title):
    print(f"\n{BOLD}{BLUE}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{RESET}")


def info(label, value):
    print(f"  {DIM}{label}:{RESET} {value}")


def success(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")


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
        # Safely read ablation type from model block, or top-level fallback
        ablation = config.get('model', {}).get('ablation', None) or config.get('ablation', None)
        
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
    
    # Set random seeds for reproducibility (Section 3.1)
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    
    banner(args.experiment)
    section("EXPERIMENT SETUP")
    info("Experiment", args.experiment)
    info("Model type", config.get('model', {}).get('type', 'dca_net'))
    info("Config", args.config or f'configs/{args.experiment}.yaml')
    info("Output dir", str(exp_manager.exp_dir))
    
    # 4. Create model
    section("MODEL")
    model = create_model(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info("Total parameters", f"{total_params:,}")
    info("Trainable parameters", f"{trainable_params:,}")
    
    # 5. Create dataloaders
    section("DATA")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    info("Train batches", str(len(train_loader)))
    info("Val batches", str(len(val_loader)))
    
    # 6. Evaluation Only Mode
    if args.evaluate_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        best_checkpoint_path = exp_manager.get_checkpoint_path('best')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            success(f"Loaded checkpoint from {best_checkpoint_path}")
        else:
            print(f"  {RED}✗ No checkpoint found to evaluate!{RESET}")
            return
            
        evaluator = Evaluator(model, test_loader, device=device, logger=logger)
        results = evaluator.evaluate(output_dir=str(exp_manager.dirs['metrics']), run_uncertainty=True)
        return
        
    # 7. Setup Training
    section("TRAINING")
    trainer = Trainer(model, config, train_loader, val_loader, logger=logger)
    
    if args.resume:
        last_checkpoint_path = exp_manager.get_checkpoint_path('last')
        if os.path.exists(last_checkpoint_path):
            trainer.load_checkpoint(last_checkpoint_path)
            success(f"Resumed from {last_checkpoint_path}")
            
    # 8. Training loop
    num_epochs = config.get('training', {}).get('num_epochs', 60)
    info("Epochs", str(num_epochs))
    info("Batch size", str(config.get('training', {}).get('batch_size', 64)))
    info("Learning rate", str(config.get('training', {}).get('learning_rate', 0.0003)))
    print()
    
    trainer.train(num_epochs=num_epochs, dry_run=args.dry_run)
    
    section("FINAL EVALUATION ON TEST SET")
    
    # Load best checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_checkpoint_path = exp_manager.get_checkpoint_path('best')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        evaluator = Evaluator(model, test_loader, device=device, logger=logger)
        test_results = evaluator.evaluate(output_dir=str(exp_manager.dirs['metrics']), run_uncertainty=True)
        
        # Save results map
        final_results = {
            'experiment': args.experiment,
            'best_epoch': checkpoint.get('epoch', 0),
            'best_val_auc': checkpoint.get('best_val_auc', 0.0),
            'test_results': test_results,
            'config': config
        }
        
        exp_manager.save_results(final_results)
        exp_manager.save_metadata()
    
    section("EXPERIMENT COMPLETE")
    success(f"Results saved to: {exp_manager.exp_dir}")
    print()

if __name__ == '__main__':
    args = parse_args()
    train_experiment(args)
