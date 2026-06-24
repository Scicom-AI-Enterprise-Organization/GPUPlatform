"""Minimal stand-in for vllm.logger.init_logger (the kernel only constructs a logger,
it never logs)."""
import logging


def init_logger(name):
    return logging.getLogger(name)
