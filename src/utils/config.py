"""
Configuration loader for DCA-Net training.
"""

import yaml
from pathlib import Path


def load_config(config_path="configs/training_config.yaml"):
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure all directories exist
    for dir_key in ['checkpoint_dir', 'log_dir']:
        if dir_key in config.get('logging', {}):
            Path(config['logging'][dir_key]).mkdir(parents=True, exist_ok=True)
    
    for dir_key in ['model_save_dir', 'log_dir', 'figures_dir']:
        if dir_key in config.get('paths', {}):
            Path(config['paths'][dir_key]).mkdir(parents=True, exist_ok=True)
    
    return config
