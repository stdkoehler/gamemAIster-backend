""" generate python logger that logs output as json """

import logging
from pythonjsonlogger import jsonlogger


def configure_logger(name: str):
    """
    Configures a logger with the specified name and sets the log level to DEBUG.
    It also adds a console handler that logs messages to the console.
    The log messages are formatted as JSON using the pythonjsonlogger.JsonFormatter class.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.

    Example:
        >>> logger = configure_logger('my_logger')
        >>> logger.debug('This is a debug message')
        >>> logger.info('This is an info message')
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
