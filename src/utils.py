"""
Logging configuration for AMC project
Provides consistent logging across all modules
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, level: int = logging.INFO, 
                 log_to_file: bool = False, log_dir: str = "logs") -> logging.Logger:
    """
    Setup a logger with consistent formatting
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        log_to_file: Whether to also log to file
        log_dir: Directory for log files
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter with colors for console
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"amc_{timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# Default logger for quick imports
def get_logger(name: str = "amc") -> logging.Logger:
    """Get or create a logger with default settings"""
    return setup_logger(name)
