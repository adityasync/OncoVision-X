#!/usr/bin/env python3
"""
Experiment Manager - Handles all experiment organization
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path

class ExperimentManager:
    """
    Manages experiment directory structure, naming, and metadata
    """
    
    def __init__(self, experiment_name, base_dir='experiments'):
        """
        Args:
            experiment_name: e.g., 'ablation_no_context'
            base_dir: Base experiments directory
        """
        self.name = experiment_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / experiment_name
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Define all subdirectories
        self.dirs = {
            'root': self.exp_dir,
            'checkpoints': self.exp_dir / 'checkpoints',
            'logs': self.exp_dir / 'logs',
            'metrics': self.exp_dir / 'metrics',
            'plots': self.exp_dir / 'plots'
        }
        
        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'status': 'initialized',
            'config': {},
            'results': {}
        }
        
        print(f"✓ Experiment initialized: {experiment_name}")
        print(f"  Directory: {self.exp_dir}")
    
    def get_checkpoint_path(self, checkpoint_type='best'):
        """
        Get path for saving checkpoints
        
        Args:
            checkpoint_type: 'best', 'last', or 'epoch_XXX'
        
        Returns:
            Path to checkpoint file
        """
        filename = f"{checkpoint_type}.pth"
        return self.dirs['checkpoints'] / filename
    
    def get_log_path(self):
        """Get path for training log"""
        filename = f"training_{self.timestamp}.log"
        return self.dirs['logs'] / filename
    
    def get_metrics_path(self, split='train'):
        """
        Get path for metrics CSV
        
        Args:
            split: 'train', 'val', or 'test'
        """
        filename = f"{split}_metrics.csv"
        return self.dirs['metrics'] / filename
    
    def get_plot_path(self, plot_name):
        """
        Get path for saving plots
        
        Args:
            plot_name: e.g., 'roc_curve', 'confusion_matrix'
        """
        filename = f"{plot_name}.png"
        return self.dirs['plots'] / filename
    
    def save_config(self, config):
        """Save experiment configuration"""
        config_path = self.exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.metadata['config'] = config
        print(f"✓ Config saved: {config_path}")
    
    def save_results(self, results):
        """Save final experiment results"""
        results_path = self.dirs['metrics'] / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.metadata['results'] = results
        self.metadata['status'] = 'completed'
        
        print(f"✓ Results saved: {results_path}")
    
    def save_metadata(self):
        """Save experiment metadata"""
        metadata_path = self.exp_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✓ Metadata saved: {metadata_path}")
    
    def load_results(self):
        """Load experiment results if exists"""
        results_path = self.dirs['metrics'] / 'final_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def list_experiments(base_dir='experiments'):
        """List all experiments in base directory"""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        experiments = []
        for exp_dir in base_path.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                metadata_path = exp_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    experiments.append(metadata)
        
        return experiments


# Example usage
if __name__ == '__main__':
    # Create experiment manager
    exp = ExperimentManager('ablation_no_context')
    
    # Save config
    config = {
        'model': 'DCANet',
        'ablation': 'no_context',
        'batch_size': 32,
        'learning_rate': 3e-4
    }
    exp.save_config(config)
    
    # Get paths for training
    print(f"Best checkpoint: {exp.get_checkpoint_path('best')}")
    print(f"Training log: {exp.get_log_path()}")
    print(f"ROC plot: {exp.get_plot_path('roc_curve')}")
