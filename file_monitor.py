from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from agents.document_rag_agent import DocumentRAGState, DocumentRAGAgent
from agents.document_parsing_agent import DocumentLayoutParsingState, DocumentParsingAgent
from pathlib import Path
import asyncio
from utils.logger import PerformanceLogger, AsyncLogger
from utils.config import LOG_LEVEL
from document_ai_agents.logger import logger
import time

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class DocumentUpdateHandler(FileSystemEventHandler):
    """
    Handle file system events for PDF updates, triggering RAG updates asynchronously.
    """
    def __init__(self, rag_agent: DocumentRAGAgent):
        self.rag_agent = rag_agent
        self.perf_logger = PerformanceLogger()

    async def on_modified(self, event):
        """
        Asynchronously handle PDF file modifications, updating the RAG knowledge base.
        
        Args:
            event: Watchdog event object containing file system event details.
        """
        if not event.src_path.endswith(".pdf") or not Path(event.src_path).is_file():
            return

        self.perf_logger.start()
        await AsyncLogger.info(f"Detected modification in PDF: {event.src_path}")
        try:
            parser = DocumentParsingAgent()
            parse_state = DocumentLayoutParsingState(document_path=event.src_path)
            parsed = await parser.graph.ainvoke(parse_state)
            
            rag_state = DocumentRAGState(
                question="Update knowledge base",
                document_path=str(Path(event.src_path)),
                pages_as_base64_jpeg_images=parsed["pages_as_base64_jpeg_images"],
                pages_as_text=parsed["pages_as_text"],
                documents=parsed["documents"]
            )
            start_time = time.time()
            await self.rag_agent.graph.ainvoke(rag_state)
            duration = time.time() - start_time
            await AsyncLogger.info(f"Updated RAG for {event.src_path} in {duration:.2f} seconds")
            await self.perf_logger.async_stop("file_monitor")
        except Exception as e:
            await AsyncLogger.error(f"Error updating RAG for {event.src_path}: {e}")
            await self.perf_logger.async_stop("file_monitor")

def start_monitoring(directory: str, polling_interval: float = 1.0):
    """
    Start monitoring a directory for PDF changes, running asynchronously.
    
    Args:
        directory (str): Directory path to monitor for PDF changes.
        polling_interval (float): Interval in seconds for polling file system changes (default: 1.0).
    """
    if not Path(directory).is_dir():
        raise ValueError(f"Directory not found or invalid: {directory}")

    rag_agent = agent_registry.get_agent("rag_agent")()
    event_handler = DocumentUpdateHandler(rag_agent)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    
    logger.info(f"Monitoring directory: {directory} with interval {polling_interval} seconds")
    try:
        while True:
            time.sleep(polling_interval)  # Keep the main thread alive
    except KeyboardInterrupt:
        observer.stop()
        logger.info("File monitoring stopped")
    observer.join()

async def async_start_monitoring(directory: str, polling_interval: float = 1.0):
    """
    Asynchronously start monitoring a directory for PDF changes.
    
    Args:
        directory (str): Directory path to monitor for PDF changes.
        polling_interval (float): Interval in seconds for polling file system changes (default: 1.0).
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, start_monitoring, directory, polling_interval)

if __name__ == "__main__":
    # Synchronous example
    start_monitoring("./documents", 1.0)

    # Asynchronous example
    async def run_async():
        await async_start_monitoring("./documents", 1.0)
    
    asyncio.run(run_async())
