from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_core.documents import Document
from document_utils import async_extract_text_from_pdf, async_extract_images_from_pdf
from utils.image_utils import pil_image_to_base64_jpeg
from utils.llm_client import LLMClient
from utils.config import LOG_LEVEL, MAX_IMAGES, IMAGE_QUALITY
from utils.logger import logger, AsyncLogger
import asyncio
from agent_registry import agent_registry  # Added import for agent_registry

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class DocumentLayoutParsingState(BaseModel):
    document_path: str
    pages_as_text: List[str] = Field(default_factory=list)
    pages_as_base64_jpeg_images: List[str] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)

@agent_registry.register(
    name="parsing_agent",
    capabilities=["parsing"],
    supported_doc_types=["generic", "term_sheet", "research_paper"]
)
class DocumentParsingAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.graph = self.build_agent()

    async def parse_document(self, state: DocumentLayoutParsingState) -> dict:
        try:
            # Extract text and images asynchronously
            text_pages = await async_extract_text_from_pdf(state.document_path)
            image_pages = await async_extract_images_from_pdf(state.document_path)
            
            # Limit images to MAX_IMAGES and convert to base64
            state.pages_as_text = text_pages[:5]  # Limit text for brevity
            state.pages_as_base64_jpeg_images = [
                await asyncio.to_thread(pil_image_to_base64_jpeg, img, quality=IMAGE_QUALITY)
                for img in image_pages[:MAX_IMAGES]
            ]
            
            # Create documents for RAG
            state.documents = [
                Document(page_content=text, metadata={"page_number": i, "element_type": "Text-block"})
                for i, text in enumerate(state.pages_as_text)
            ] + [
                Document(page_content="", metadata={"page_number": i, "element_type": "Image", "image_base64": img_base64})
                for i, img_base64 in enumerate(state.pages_as_base64_jpeg_images)
            ]
            
            await AsyncLogger.info(f"Parsed document {state.document_path} into {len(state.documents)} documents")
            return {
                "pages_as_text": state.pages_as_text,
                "pages_as_base64_jpeg_images": state.pages_as_base64_jpeg_images,
                "documents": state.documents
            }
        except Exception as e:
            await AsyncLogger.error(f"Error parsing document {state.document_path}: {e}")
            return {
                "pages_as_text": [],
                "pages_as_base64_jpeg_images": [],
                "documents": []
            }

    def build_agent(self):
        builder = StateGraph(DocumentLayoutParsingState)
        builder.add_node("parse_document", self.parse_document)
        builder.add_edge(START, "parse_document")
        builder.add_edge("parse_document", END)
        return builder.compile()
