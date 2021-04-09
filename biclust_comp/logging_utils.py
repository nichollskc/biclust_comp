import logging
import sys
import traceback

verbosity_to_logging_level = {
    0:  logging.CRITICAL,
    1:  logging.ERROR,
    2:  logging.WARNING,
    3:  logging.INFO,
    4:  logging.DEBUG,
}


def exception_handler(etype, value, tb):
    # Report to file
    logging.exception(''.join(traceback.format_exception(etype, value, tb)))
    # Also use default exception handling steps
    sys.__excepthook__(etype, value, tb)


def setup_logging(verbosity, logfile=None):
    """Initialises logging with custom formatting and a given verbosity level."""
    format_string = "%(asctime)s %(levelname)s | %(module)s - %(funcName)s:%(lineno)d: %(message)s"
    level = verbosity_to_logging_level[verbosity]

    log_formatter = logging.Formatter(format_string)
    root_logger = logging.getLogger()

    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Set up logging to console with verbosity level {logging.getLevelName(level)}")

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Set up logging to file {logfile} "
                     f"with verbosity level {logging.getLevelName(level)}")

    # Install exception handler
    sys.excepthook = exception_handler

def get_example_element(list_or_set):
    # Try to extract an element from the set or list, by creating an iterable (iter)
    #   and then calling next on the iterable to extract the first element
    #   This avoids damaging the original list or set.
    try:
        element = next(iter(list_or_set))
    # If there are no elements, a StopIteration exception will be raised, in
    #   which case we just log a little error message to note that there are no elements
    except StopIteration:
        element = "**No elements in list!**"
    return element
