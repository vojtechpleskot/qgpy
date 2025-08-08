"""
utils.py
========
Utility functions for the qgpy package.

Functions
---------
create_logger(name: str, outdir: Optional[str] = None) -> logging.Logger
    Create a logger instance with a file handler that writes logs to a specified directory.
"""

import logging
from typing import Optional

def create_logger(name: str, level: str = "INFO", outdir: Optional[str] = None) -> logging.Logger:
    """
    Create a logger instance with a file handler that writes logs to a specified directory.

    Parameters
    ----------
    name : str
        The name of the logger.
    outdir : Optional[str]
        The output directory where the log file will be stored. If None, logs will not be written to a file.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    
    # Create a logger instance with the specified name.
    logger = logging.getLogger(name)

    # Convert the level to an integer.
    try:
        level_int = getattr(logging, level.upper())
    except:
        raise ValueError(f"Invalid logging level: {level}. Use one of the following strings: DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    
    # Set the logging level.
    logger.setLevel(level_int)

    # Create a file handler if an output directory is specified.
    # If outdir is None, the logger will not write to a file.
    if outdir:
        file_handler = logging.FileHandler(f'{outdir}/{name}.log')
        file_handler.setLevel(level_int)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger