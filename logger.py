from prometheus_client import Counter, Histogram
from typing import Optional
import time
import asyncio
from loguru import logger
from utils.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE

# Set log level from config
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format=LOG_FORMAT)
if LOG_FILE:
    logger.add(LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT, rotation="500 MB", retention="10 days")

class PerformanceLogger:
    """
    A class to log performance metrics (e.g., request counts, response times) using Prometheus.
    """
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize the PerformanceLogger with an optional CollectorRegistry.
        
        Args:
            registry (Optional[CollectorRegistry]): Prometheus CollectorRegistry to use for metrics.
        """
        self.registry = registry or CollectorRegistry()  # Use provided registry or default
        self.start_time = None
        self.requests = Counter("api_requests_total", "Total API requests by endpoint", ["endpoint"], registry=self.registry)
        self.response_time = Histogram("api_response_time_seconds", "API response time in seconds by endpoint", ["endpoint"], registry=self.registry)

    def start(self):
        """Start timing a request."""
        self.start_time = time.time()

    async def async_stop(self, endpoint: str):
        """
        Asynchronously stop timing a request and log the duration.
        
        Args:
            endpoint (str): The endpoint being measured.
        """
        if self.start_time is None:
            logger.warning(f"No start time recorded for endpoint {endpoint}")
            return
        duration = time.time() - self.start_time
        self.response_time.labels(endpoint=endpoint).observe(duration)
        logger.info(f"Endpoint {endpoint} processed in {duration:.2f} seconds")
        self.start_time = None

    def stop(self, endpoint: str):
        """
        Synchronously stop timing a request and log the duration.
        
        Args:
            endpoint (str): The endpoint being measured.
        """
        if self.start_time is None:
            logger.warning(f"No start time recorded for endpoint {endpoint}")
            return
        duration = time.time() - self.start_time
        self.response_time.labels(endpoint=endpoint).observe(duration)
        logger.info(f"Endpoint {endpoint} processed in {duration:.2f} seconds")
        self.start_time = None

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
