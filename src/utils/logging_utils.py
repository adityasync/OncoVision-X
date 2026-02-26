"""
Logging utilities for DCA-Net training.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(config, name="dca-net"):
    """Setup logging with file + console handlers.
    
    Args:
        config: Configuration dictionary
        name: Logger name
        
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path(config.get('paths', {}).get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"training_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger
