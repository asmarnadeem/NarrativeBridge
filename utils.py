"""Lightweight utilities used across train/eval scripts."""

import logging


def get_logger(filename=None):
    """Configure root logger with stream + optional file handler.

    Mirrors the behaviour of the original `feature_extractor.util.get_logger`
    but lives in a more sensible location now that the graph feature-extractor
    code has been removed.
    """
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)
    return logger
