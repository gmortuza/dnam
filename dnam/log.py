import logging
import sys


def get_logger(verbose=0, logger_name=__name__):
    """
    Format the logger to show the debug information
    :param: verbose: 0 -> Print only error. This is the default.
                     1 -> Print debug, information, warning and error
                     2 -> Print information, warning and error
                     3 -> Print warning and error
    :return: Logger
    """
    logger = logging.getLogger(logger_name)
    stream_handler = logging.StreamHandler(sys.stdout)
    if verbose == 1:
        stream_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif verbose == 2:
        stream_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    elif verbose == 3:
        stream_handler.setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)
    else:
        stream_handler.setLevel(logging.ERROR)
        logger.setLevel(logging.ERROR)

    stream_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(stream_handler)

    return logger
