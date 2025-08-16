import logging
import os
from logging.handlers import RotatingFileHandler
from .models import Loggers

# Constants for log rotation settings
LOG_DIR = 'logs'
LOG_MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3                  # Keep 3 old log files

def setup_logger(name: str, level: int, formatter_str: str,
                 console_output: bool = False, file_path: str = None, use_rotation: bool = False):
    """
    A versatile function to configure and retrieve a logger instance.

    :param name: A unique name for the logger.
    :param level: The minimum logging level for the logger.
    :param formatter_str: The format string for log messages.
    :param console_output: Whether to enable console output.
    :param file_path: The full path to the log file (if file output is enabled).
    :param use_rotation: Whether to enable log file rotation.
    :return: A configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent logs from being passed to the root logger

    # Clear previous handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(formatter_str, datefmt='%Y-%m-%d %H:%M:%S')

    # Configure console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Configure file handler
    if file_path:
        if use_rotation:
            handler = RotatingFileHandler(
                file_path,
                maxBytes=LOG_MAX_SIZE_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
        else:
            handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def initialize_loggers() -> Loggers:
    """
    Initializes and returns a container with three distinct loggers:
    1. file: Logs DEBUG level and above to a detailed file.
    2. console: Logs INFO level and above to the console.
    3. combined: Logs INFO level and above to both a file and the console.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. File-only logger for detailed debug information
    file_logger = setup_logger(
        name='FileLogger',
        level=logging.DEBUG,
        file_path=os.path.join(LOG_DIR, 'parser_details.log'),
        formatter_str='%(asctime)s - %(levelname)s - %(message)s',
        use_rotation=True
    )

    # 2. Console-only logger for high-level user feedback
    console_logger = setup_logger(
        name='ConsoleLogger',
        level=logging.INFO,
        console_output=True,
        formatter_str='%(asctime)s - %(levelname)-8s - %(message)s'
    )

    # 3. Combined logger for a general overview log file
    combined_logger = setup_logger(
        name='CombinedLogger',
        level=logging.INFO,
        console_output=True,
        file_path=os.path.join(LOG_DIR, 'combined.log'),
        formatter_str='%(asctime)s - %(levelname)-8s - %(message)s',
        use_rotation=True
    )
    
    return Loggers(file=file_logger, console=console_logger, combined=combined_logger)