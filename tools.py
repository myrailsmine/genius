from functools import lru_cache, wraps
from typing import Any, Callable, Optional, AsyncGenerator
import requests
import wikipedia
from duckduckgo_search import DDGS
from pydantic import BaseModel
from document_ai_agents.logger import logger
from utils.logger import PerformanceLogger, AsyncLogger
from utils.config import LOG_LEVEL, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES
from utils.llm_client import LLMClient
import asyncio
import time
from html.parser import HTMLParser

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class ErrorResponse(BaseModel):
    error: str
    success: bool = False

class PageSummary(BaseModel):
    page_title: str
    page_summary: str
    page_url: str

class SearchResponse(BaseModel):
    page_summaries: list[PageSummary]

class FullPage(BaseModel):
    page_title: str
    page_url: str
    content: str

def catch_exceptions(func: Callable) -> Callable:
    """
    Decorator to catch exceptions in tool functions and return an ErrorResponse.
    
    Args:
        func (Callable): The tool function to wrap.
    
    Returns:
        Callable: Wrapped function that handles exceptions.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        perf_logger = PerformanceLogger()
        perf_logger.start()
        try:
            response = await func(*args, **kwargs)
            await AsyncLogger.info(f"Tool {func.__name__} executed successfully for query: {args[0] if args else kwargs.get('query', 'No query')}")
            await perf_logger.async_stop(f"tool_{func.__name__}")
            return response
        except Exception as e:
            await AsyncLogger.error(f"Tool {func.__name__} failed: {e}")
            await perf_logger.async_stop(f"tool_{func.__name__}")
            return ErrorResponse(error=str(e))
    return wrapper

class HTMLStripper(HTMLParser):
    """
    A simple HTML parser to strip HTML tags and return plain text.
    """
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return "".join(self.text)

@lru_cache(maxsize=1024)
def cached_wikipedia_page(title: str) -> wikipedia.WikipediaPage:
    """
    Cached Wikipedia page retrieval to improve performance.
    
    Args:
        title (str): Title of the Wikipedia page.
    
    Returns:
        wikipedia.WikipediaPage: Cached Wikipedia page object.
    """
    return wikipedia.page(title=title, auto_suggest=False)

@catch_exceptions
async def search_wikipedia(search_query: str, max_results: int = 5) -> SearchResponse:
    """
    Asynchronously search Wikipedia and return page summaries.
    
    Args:
        search_query (str): Query to search on Wikipedia.
        max_results (int): Maximum number of results to return (default: 5).
    
    Returns:
        SearchResponse: Pydantic model with list of page summaries.
    """
    titles = await asyncio.to_thread(wikipedia.search, search_query, results=max_results)
    page_summaries = []
    for title in titles[:max_results]:
        try:
            page = cached_wikipedia_page(title)
            page_summary = PageSummary(
                page_title=page.title,
                page_summary=page.summary,
                page_url=page.url
            )
            page_summaries.append(page_summary)
        except (wikipedia.DisambiguationError, wikipedia.PageError) as e:
            await AsyncLogger.warning(f"Error getting Wikipedia page {title}: {e}")
    return SearchResponse(page_summaries=page_summaries)

@catch_exceptions
async def get_wikipedia_page(page_title: str, max_text_size: int = 16_000) -> FullPage:
    """
    Asynchronously retrieve a Wikipedia page's full content.
    
    Args:
        page_title (str): Title of the Wikipedia page.
        max_text_size (int): Maximum size of text content to return (default: 16,000).
    
    Returns:
        FullPage: Pydantic model with page title, URL, and content.
    """
    try:
        page = cached_wikipedia_page(page_title)
        stripper = HTMLStripper()
        stripper.feed(page.html())
        full_content = stripper.get_text()
        full_page = FullPage(
            page_title=page.title,
            page_url=page.url,
            content=full_content[:max_text_size],
        )
    except (wikipedia.DisambiguationError, wikipedia.PageError) as e:
        await AsyncLogger.warning(f"Error getting Wikipedia page {page_title}: {e}")
        full_page = FullPage(
            page_title=page_title,
            page_url="",
            content="",
        )
    return full_page

@catch_exceptions
async def search_duck_duck_go(search_query: str, max_results: int = 10) -> SearchResponse:
    """
    Asynchronously search DuckDuckGo and return page summaries.
    
    Args:
        search_query (str): Query to search on DuckDuckGo.
        max_results (int): Maximum number of results to return (default: 10).
    
    Returns:
        SearchResponse: Pydantic model with list of page summaries.
    """
    with DDGS() as dd:
        results_generator = await asyncio.to_thread(dd.text, search_query, max_results=max_results, backend="api")
        return SearchResponse(
            page_summaries=[
                PageSummary(
                    page_title=x["title"],
                    page_summary=x["body"],
                    page_url=x["href"]
                )
                for x in results_generator
            ]
        )

@catch_exceptions
async def get_page_content(page_url: str, max_text_size: int = 16_000) -> FullPage:
    """
    Asynchronously retrieve content from a web page.
    
    Args:
        page_url (str): URL of the web page.
        max_text_size (int): Maximum size of text content to return (default: 16,000).
    
    Returns:
        FullPage: Pydantic model with page title, URL, and content.
    """
    try:
        response = await asyncio.to_thread(requests.get, page_url)
        response.raise_for_status()
        html = response.text
        stripper = HTMLStripper()
        stripper.feed(html)
        content = stripper.get_text()
        content = "\n".join([x for x in content.split("\n") if x.strip()])
        # Extract title from HTML (simplified)
        title = html.split("<title>")[1].split("</title>")[0] if "<title>" in html else page_url
        return FullPage(
            page_title=title,
            page_url=page_url,
            content=content[:max_text_size],
        )
    except Exception as e:
        await AsyncLogger.warning(f"Error getting page content from {page_url}: {e}")
        return FullPage(page_title=page_url, page_url=page_url, content="")

@catch_exceptions
async def llm_enhanced_tool(query: str, multimodal_context: Optional[dict] = None) -> str:
    """
    Use an LLM to enhance tool output with multimodal context.
    
    Args:
        query (str): Query or task for the tool.
        multimodal_context (Optional[dict]): Dictionary with 'text' and 'images' (base64 strings) for multimodal input.
    
    Returns:
        str: Enhanced response from the LLM, considering multimodal data.
    """
    llm = LLMClient()
    text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
    image_context = "\n".join([await asyncio.to_thread(llm.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

    prompt = (
        f"Enhance this query with additional context: {query}\n"
        f"Text Context: {text_context}\n"
        f"Image Context: {image_context}\n"
        "Provide a detailed, concise response combining the query and context:"
    )
    start_time = time.time()
    response = await asyncio.to_thread(llm.generate, prompt, image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0], max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP_SEQUENCES)
    duration = time.time() - start_time
    await AsyncLogger.info(f"LLM-enhanced tool response generated in {duration:.2f} seconds: {response}")
    return response

async def stream_tool_output(query: str, tool: Callable, multimodal_context: Optional[dict] = None) -> AsyncGenerator[str, None]:
    """
    Stream tool output token by token for real-time display.
    
    Args:
        query (str): Query or task for the tool.
        tool (Callable): Tool function to execute.
        multimodal_context (Optional[dict]): Dictionary with 'text' and 'images' (base64 strings) for multimodal input.
    
    Yields:
        str: Tokens of the tool output as they are generated.
    """
    perf_logger = PerformanceLogger()
    perf_logger.start()
    try:
        result = await tool(query, multimodal_context)
        if isinstance(result, (SearchResponse, FullPage)):
            yield str(result.model_dump_json()) + "\n"
        elif isinstance(result, ErrorResponse):
            yield str(result.model_dump_json()) + "\n"
        else:
            yield f"Processing: {query}\n"
            for token in str(result).split():
                yield token + " "
                await asyncio.sleep(0.05)  # Simulate streaming delay
            yield "\n"
        await AsyncLogger.info(f"Streamed tool output for query: {query}")
        await perf_logger.async_stop(f"tool_{tool.__name__}")
    except Exception as e:
        await AsyncLogger.error(f"Error streaming tool output for {query}: {e}")
        yield str(ErrorResponse(error=str(e)).model_dump_json()) + "\n"
        await perf_logger.async_stop(f"tool_{tool.__name__}")
