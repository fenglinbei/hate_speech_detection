import os
import sys
import logging
from datetime import datetime
from pprint import pformat
from types import SimpleNamespace
import inspect

try:
    from loguru import logger
    HAS_LOGURU = True
except ModuleNotFoundError:
    HAS_LOGURU = False
    class _StdLogger:
        def __init__(self):
            self._logger = logging.getLogger("hateSpeechDetection")

        def remove(self):
            return None

        def add(self, *args, **kwargs):
            return None

        def level(self, level_name):
            return SimpleNamespace(name=level_name)

        def opt(self, *args, **kwargs):
            return self

        def log(self, level, message, *args, **kwargs):
            self._logger.log(getattr(logging, str(level), logging.INFO), self._format(message, *args))

        def debug(self, message, *args, **kwargs):
            self._logger.debug(self._format(message, *args))

        def info(self, message, *args, **kwargs):
            self._logger.info(self._format(message, *args))

        def warning(self, message, *args, **kwargs):
            self._logger.warning(self._format(message, *args))

        def error(self, message, *args, **kwargs):
            self._logger.error(self._format(message, *args))

        def exception(self, message, *args, **kwargs):
            self._logger.exception(self._format(message, *args))

        @staticmethod
        def _format(message, *args):
            if not args:
                return str(message)
            try:
                return str(message).format(*args)
            except Exception:
                return " ".join([str(message), *map(str, args)])

    logger = _StdLogger()

LOG_PATH = os.getenv("LOG_PATH", "logs/")
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
CONSOLE_FORMAT = '<level>{level: <8}</level>  <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>'
FILE_FORMAT = '{level: <8}  {time:YYYY-MM-DD HH:mm:ss.SSS} - {name}:{function} - {message}'

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = inspect.currentframe().f_back
        depth = 0
        while frame and logging.__file__ in frame.f_code.co_filename:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def init_logger(level: str = "INFO", log_path: str = LOG_PATH, show_console: bool = True, record_levels: list = LOG_LEVELS):
    if not HAS_LOGURU:
        logging.basicConfig(
            level=getattr(logging, str(level).upper(), logging.INFO),
            format="%(levelname)-8s %(asctime)s - %(name)s:%(funcName)s - %(message)s",
            force=True,
        )
        logger.info("Logger initialized with stdlib fallback")
        return logger

    logger.remove()  # 移除Loguru的默认处理器
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_path, current_time)

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create log directory: {e}")
        raise

    # 拦截所有标准日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("uvicorn.access").propagate = False

    # 控制台日志
    if show_console:
        logger.add(sys.stdout, level=level, format=CONSOLE_FORMAT)

    # 文件日志
    for log_level in record_levels:
        logger.add(
            os.path.join(log_dir, f'{log_level.lower()}.log'),
            encoding="utf-8",
            level=log_level,
            filter=lambda record, lvl=log_level: record["level"].name == lvl,
            format=FILE_FORMAT,
            rotation="10 MB",
            retention="7 days"
        )

    logger.info("Logger initialized in directory: {}", log_dir)
    return logger
