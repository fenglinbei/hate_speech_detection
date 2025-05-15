import os
import sys
import logging
from datetime import datetime
from pprint import pformat
from loguru import logger
from config import SETTINGS

LOG_FORMAT = '<level>{level: <8}</level>  <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>'

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def format_record(record: dict) -> str:
    format_string = LOG_FORMAT

    if record["extra"].get("payload") is not None:
        record["extra"]["payload"] = pformat(
            record["extra"]["payload"], indent=4, compact=True, width=88
        )
        format_string += "\n<level>{extra[payload]}</level>"

    format_string += "{exception}\n"
    return format_string


def init_logger(level: str = "INFO", show: bool = False):
    # Create a new log directory with the current server time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(SETTINGS.log_path, current_time)
    os.makedirs(log_dir, exist_ok=True)

    logging.getLogger().handlers = [InterceptHandler()]
    logger.configure(
        handlers=[{"sink": sys.stdout, "level": logging.DEBUG, "format": format_record}])
    
    # Remove any existing handlers, in case this is not the first call
    if not show:
        logger.remove(handler_id=None)
    
    # INFO log file
    logger.add(
        os.path.join(log_dir, 'info.log'), 
        encoding="utf-8", 
        level=level,
        filter=lambda record: record["level"].name == "INFO"
    )

    # DEBUG log file
    logger.add(
        os.path.join(log_dir, 'debug.log'), 
        encoding="utf-8", 
        level=level,
        filter=lambda record: record["level"].name == "DEBUG"
    )

    # ERROR log file
    logger.add(
        os.path.join(log_dir, 'error.log'), 
        encoding="utf-8", 
        level=level,
        filter=lambda record: record["level"].name == "ERROR"
    )

    # CRITICAL log file
    logger.add(
        os.path.join(log_dir, 'critical.log'), 
        encoding="utf-8", 
        level=level,
        filter=lambda record: record["level"].name == "CRITICAL"
    )
    
    logger.info("Logger initialized in new directory: {}", log_dir)
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    return logger