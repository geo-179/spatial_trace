"""
Logging configuration for spatial reasoning.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "spatial_trace",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "spatial_trace") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str, logger_name: str = "spatial_trace") -> None:
    """
    Set logging level for a specific logger.
    
    Args:
        level: New logging level
        logger_name: Name of the logger to update
    """
    logger = logging.getLogger(logger_name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def configure_root_logger(level: str = "WARNING") -> None:
    """
    Configure the root logger to reduce noise from third-party libraries.
    
    Args:
        level: Logging level for root logger
    """
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    root_logger.setLevel(numeric_level)


# Initialize the main logger
main_logger = setup_logger("spatial_trace", level="INFO")

# Configure root logger to reduce third-party noise
configure_root_logger("WARNING") 