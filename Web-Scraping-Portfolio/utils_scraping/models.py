import logging
from dataclasses import dataclass

@dataclass(frozen=True)
class Loggers:
    """
    A dataclass to act as a container for different logger instances.
    Using `frozen=True` makes instances of this class immutable, which is a
    good practice for container objects that shouldn't be changed after creation.
    """
    file: logging.Logger
    console: logging.Logger
    combined: logging.Logger