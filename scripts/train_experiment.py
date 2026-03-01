#!/usr/bin/env python3
"""
Universal Training Script for All Experiments
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_manager import ExperimentManager
from utils.metrics_tracker import MetricsTracker
# Import your models, datasets, etc.

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
    model_type = config['model']['type']
    
    if model_type == 'dca_net':
        ablation = config.get('ablation', None)
        from models.dca_net import DCANet
        model = DCANet(config, ablation=ablation)
    
    elif model_type == 'resnet3d18':
        from models.baselines import ResNet3D18
        model = ResNet3D18(config)
    
    elif model_type == 'resnet2d18':
        from models.baselines import ResNet2D18
        model = ResNet2D18(config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_experiment(args):
    """Main training function"""
    
    # 1. Setup experiment manager
    exp_manager = ExperimentManager(args.experiment)
    
    # 2. Load configuration
    config = load_config(args.experiment, args.config)
    exp_manager.save_config(config)
    
    # 3. Setup logging
    log_path = exp_manager.get_log_path()
    # Setup your logger here
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {args.experiment}")
    print(f"{'='*60}\n")
    
    # 4. Create model
    print("Creating model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # 5. Setup device and DataParallel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    # 6. Create dataloaders
    print("\nCreating dataloaders...")
    # Your dataloader creation logic here
    # train_loader = create_dataloader(config, 'train')
    # val_loader = create_dataloader(config, 'val')
    # test_loader = create_dataloader(config, 'test')
    
    # 7. Setup optimizer and loss
    print("\nSetting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Your loss function
    # criterion = create_loss_function(config)
    
    # 8. Setup metrics tracker
    # Commented out as MetricsTracker is a placeholder
    # metrics_tracker = MetricsTracker(exp_manager)
    
    # 9. Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}\n")
    
    best_auc = 0.0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train epoch
        # train_metrics = train_epoch(
        #     model, train_loader, optimizer, criterion, device, config
        # )
        
        # Validate epoch
        # val_metrics = validate_epoch(
        #     model, val_loader, criterion, device
        # )
        
        # Track metrics
        # metrics_tracker.update(epoch, train_metrics, val_metrics)
        
        # Save best checkpoint
        # if val_metrics['auc'] > best_auc:
        #     best_auc = val_metrics['auc']
        #     checkpoint_path = exp_manager.get_checkpoint_path('best')
        #     save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
        #     print(f"✓ New best AUC: {best_auc:.4f} - saved to {checkpoint_path}")
        
        # Save last checkpoint
        # checkpoint_path = exp_manager.get_checkpoint_path('last')
        # save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
        
        # Print summary
        # print(f"  Train Loss: {train_metrics['loss']:.4f}")
        # print(f"  Val Loss: {val_metrics['loss']:.4f}")
        # print(f"  Val AUC: {val_metrics['auc']:.4f}")
    
    # 10. Final evaluation on test set
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    
    # Load best checkpoint
    # best_checkpoint_path = exp_manager.get_checkpoint_path('best')
    # checkpoint = torch.load(best_checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    # test_results = evaluate_model(
    #     model, test_loader, device, exp_manager
    # )
    
    # Save results
    # final_results = {
    #     'experiment': args.experiment,
    #     'best_epoch': checkpoint['epoch'],
    #     'best_val_auc': checkpoint['val_auc'],
    #     'test_results': test_results,
    #     'config': config
    # }
    
    # exp_manager.save_results(final_results)
    # exp_manager.save_metadata()
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    # print(f"Best Val AUC: {checkpoint['val_auc']:.4f}")
    # print(f"Test AUC: {test_results['auc']:.4f}")
    # print(f"Test Sensitivity: {test_results['sensitivity']:.4f}")
    # print(f"Test Specificity: {test_results['specificity']:.4f}")
    print(f"\nResults saved to: {exp_manager.exp_dir}")
    

def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint"""
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': metrics['auc'],
        'val_loss': metrics['loss']
    }, path)


if __name__ == '__main__':
    args = parse_args()
    train_experiment(args)
