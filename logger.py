import time
import asyncio
from loguru import logger
from utils.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE

# Set log level from config
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format=LOG_FORMAT)
if LOG_FILE:
    logger.add(LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT, rotation="500 MB", retention="10 days")

class AsyncLogger:
    """
    A utility class for asynchronous logging with loguru.
    """
    @staticmethod
    async def info(message: str):
        """Asynchronously log an info message."""
        await asyncio.to_thread(logger.info, message)

    @staticmethod
    async def warning(message: str):
        """Asynchronously log a warning message."""
        await asyncio.to_thread(logger.warning, message)

    @staticmethod
    async def error(message: str):
        """Asynchronously log an error message."""
        await asyncio.to_thread(logger.error, message)
