import inspect
import logging
import time
from loguru import logger
from typing import Optional
from prometheus_client import Counter, Histogram
import asyncio

class InterceptHandler(logging.Handler):
    """
    Intercept standard logging calls and redirect them to loguru.
    
    Args:
        record (logging.LogRecord): The logging record to emit.
    """
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.INFO:
            return
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

class PerformanceLogger:
    """
    Log performance metrics for agents, including response time and request counts, using Prometheus.
    """
    def __init__(self):
        self.requests = Counter("requests_total", "Total requests by agent", ["agent"])
        self.response_time = Histogram("response_time_seconds", "Response time in seconds by agent", ["agent"])
        self.start_time = None

    def start(self) -> None:
        """Start timing a request."""
        self.start_time = time.time()

    def stop(self, agent_name: str) -> None:
        """Stop timing and log the duration, incrementing metrics."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.response_time.labels(agent=agent_name).observe(duration)
            self.requests.labels(agent=agent_name).inc()
            logger.info(f"Agent {agent_name} processed in {duration:.2f} seconds")

    async def async_stop(self, agent_name: str) -> None:
        """Asynchronously stop timing and log the duration, incrementing metrics."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.response_time.labels(agent=agent_name).observe(duration)
            self.requests.labels(agent=agent_name).inc()
            logger.info(f"Agent {agent_name} processed in {duration:.2f} seconds")

class AsyncLogger:
    """
    Async wrapper for logging to ensure compatibility with asynchronous workflows.
    """
    @staticmethod
    async def info(message: str, *args, **kwargs) -> None:
        """Asynchronously log an info message."""
        await asyncio.to_thread(logger.info, message, *args, **kwargs)

    @staticmethod
    async def error(message: str, *args, **kwargs) -> None:
        """Asynchronously log an error message."""
        await asyncio.to_thread(logger.error, message, *args, **kwargs)

    @staticmethod
    async def warning(message: str, *args, **kwargs) -> None:
        """Asynchronously log a warning message."""
        await asyncio.to_thread(logger.warning, message, *args, **kwargs)

# Configure standard logging to use loguru
logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG, force=True)

# Initialize global performance logger
performance_logger = PerformanceLogger()

# Example usage in synchronous context
def log_agent_performance(agent_name: str, message: str) -> None:
    """
    Log agent performance in a synchronous context.
    
    Args:
        agent_name (str): Name of the agent.
        message (str): Message to log.
    """
    performance_logger.start()
    logger.info(f"Starting {agent_name}: {message}")
    # Simulate work
    time.sleep(1)
    performance_logger.stop(agent_name)
    logger.info(f"Completed {agent_name}: {message}")

# Example usage in asynchronous context
async def async_log_agent_performance(agent_name: str, message: str) -> None:
    """
    Log agent performance in an asynchronous context.
    
    Args:
        agent_name (str): Name of the agent.
        message (str): Message to log.
    """
    performance_logger.start()
    await AsyncLogger.info(f"Starting {agent_name}: {message}")
    # Simulate async work
    await asyncio.sleep(1)
    await performance_logger.async_stop(agent_name)
    await AsyncLogger.info(f"Completed {agent_name}: {message}")

if __name__ == "__main__":
    # Synchronous example
    log_agent_performance("rag_agent", "Processing document QA")

    # Asynchronous example
    async def run_async():
        await async_log_agent_performance("rag_agent", "Processing multimodal QA")
    
    asyncio.run(run_async())
