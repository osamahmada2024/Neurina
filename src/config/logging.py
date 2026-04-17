import logging

from .settings import settings


_NOISY_LOGGER_LEVELS = {
    "urllib3": logging.ERROR,
    "urllib3.connectionpool": logging.ERROR,
    "urllib3.util.retry": logging.ERROR,
    "uvicorn.access": logging.WARNING,
    "uvicorn.error": logging.WARNING,
    "watchfiles": logging.WARNING,
}


def _resolve_level(level_name: str) -> int:
    return getattr(logging, str(level_name).upper(), logging.WARNING)


def configure_logging() -> None:
    """Keep runtime logs short and suppress noisy third-party retries."""
    app_level = _resolve_level(settings.APP_LOG_LEVEL)
    feedback_level = _resolve_level(settings.APP_FEEDBACK_LOG_LEVEL)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=app_level,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    else:
        root_logger.setLevel(app_level)

    logging.getLogger("src").setLevel(feedback_level)

    for logger_name, min_level in _NOISY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(max(app_level, min_level))
