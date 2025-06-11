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
    logger = logging.getLogger(name)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    logger.handlers.clear()

    if format_string is None:
        format_string = "%(message)s"

    formatter = logging.Formatter(format_string)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

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


main_logger = setup_logger("spatial_trace", level="INFO")

configure_root_logger("WARNING")
