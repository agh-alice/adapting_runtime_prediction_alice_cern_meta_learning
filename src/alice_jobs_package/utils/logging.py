"""Logging utilities."""

import logging
import os

from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    WARN,
    WARNING,
)
from typing import Optional


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_default_logging_level():
    env_level_str = os.getenv("ALICE_JOBS_PACKAGE_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option ALICE_JOBS_PACKAGE_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    library_root_logger = _get_library_root_logger()
    library_root_logger.addHandler(logging.StreamHandler())
    library_root_logger.setLevel(_get_default_logging_level())


def _reset_library_root_logger() -> None:
    library_root_logger = _get_library_root_logger()
    library_root_logger.setLevel(logging.NOTSET)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = _get_library_name()
    return logging.getLogger(name)


def get_verbosity() -> int:
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    return set_verbosity(INFO)


def set_verbosity_warning():
    return set_verbosity(WARNING)


def set_verbosity_debug():
    return set_verbosity(DEBUG)


def set_verbosity_error():
    return set_verbosity(ERROR)


def disable_propagation() -> None:
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    _get_library_root_logger().propagate = True


# Configure the library root logger at the module level (singleton-like)
_configure_library_root_logger()