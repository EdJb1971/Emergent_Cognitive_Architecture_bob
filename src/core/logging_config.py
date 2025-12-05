import logging
from pythonjsonlogger import jsonlogger
from src.core.config import settings

def setup_logging():
    """
    Sets up structured logging for the application using python-json-logger.
    Logs are output in JSON format.
    """
    log_level = settings.LOG_LEVEL.upper()
    numeric_log_level = getattr(logging, log_level, logging.INFO)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)

    # Clear existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(levelname)s %(asctime)s %(filename)s %(lineno)d %(message)s',
        json_ensure_ascii=False
    )

    # Create a console handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Configure uvicorn loggers to use the same handler and level
    # This ensures uvicorn's access and error logs are also structured
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(numeric_log_level)
    uvicorn_access_logger.propagate = False # Prevent logs from going to root logger twice
    if uvicorn_access_logger.hasHandlers():
        uvicorn_access_logger.handlers.clear()
    uvicorn_access_logger.addHandler(handler)

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.setLevel(numeric_log_level)
    uvicorn_error_logger.propagate = False
    if uvicorn_error_logger.hasHandlers():
        uvicorn_error_logger.handlers.clear()
    uvicorn_error_logger.addHandler(handler)

    # Set the log level for other libraries that might be chatty
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    root_logger.info(f"Logging configured with level: {log_level}")