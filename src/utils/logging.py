"""Logging utilities with Rich."""
import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

console = Console()
_logger = None


def setup_logger(
    name: str = "mm_reversal",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: int = 0
) -> logging.Logger:
    """Setup logger with Rich formatting."""
    global _logger
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    if rank == 0:
        console_handler = RichHandler(console=console, show_time=True, show_path=False)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    if _logger is None:
        return setup_logger()
    return _logger
