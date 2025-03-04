from agent_registry import agent_registry, BaseAgentState
from langgraph.graph import StateGraph, Send
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator
from utils.llm_client import LLMClient
from utils.config import LOG_LEVEL, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES
from document_ai_agents.logger import logger
import asyncio
import json

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class NewAgentState(BaseModel):
    messages: List[dict] = Field(default_factory=list)
    document_path: Optional[str] = None
    question: Optional[str] = None
    summarize: bool = False
    documents: List[Document] = Field(default_factory=list)
    multimodal_context: Optional[dict] = Field(None, description="Text and image data from documents")  # For multimodal support

@agent_registry.register(
    name="new_agent",
    capabilities=["qa", "summarization", "multimodal"],
    supported_doc_types=["generic"]
)
class NewAgent:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.llm_client = LLMClient()
        self.embedder = SentenceTransformer(embed_model)
        self.graph = self.build_agent()

    async def preprocess_document(self, state: NewAgentState):
        if not state.document_path or not Path(state.document_path).is_file() or not state.document_path.endswith('.pdf'):
            return state
        text_pages = await asyncio.to_thread(extract_text_from_pdf, state.document_path)
        state.documents = [
            Document(page_content=page, metadata={"page_number": i, "document_path": state.document_path})
            for i, page in enumerate(text_pages)
        ]
        
        # Extract multimodal context (text + images) using DocumentParsingAgent
        parser = agent_registry.get_agent("parsing_agent")()
        parse_state = DocumentLayoutParsingState(document_path=state.document_path)
        parsed = await parser.graph.ainvoke(parse_state)
        state.multimodal_context = {
            "text": parsed["pages_as_text"],
            "images": parsed["pages_as_base64_jpeg_images"]
        }
        return state

    async def process_message(self, state: NewAgentState) -> dict:
        prompt = "\n".join([m["parts"][0] if isinstance(m["parts"][0], str) else m["parts"][0]["text"] for m in state.messages])
        if state.question or state.summarize:
            text_context = "\n".join(state.multimodal_context.get("text", [])[:500]) if state.multimodal_context else ""
            image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in state.multimodal_context.get("images", [])[:5]]) if state.multimodal_context else ""
            task = "summarize" if state.summarize else "answer"
            prompt = (
                f"{task.capitalize()} the following: {state.question or 'this document'}\n"
                f"Text Context: {text_context}\n"
                f"Image Context: {image_context}\n"
                "Provide a detailed response:"
            )
        
        # Use agentic reasoning for structured response
        response = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not state.multimodal_context or not state.multimodal_context["images"] else state.multimodal_context["images"][0])
        return {"messages": state.messages + [{"role": "system", "parts": [response]}]}

    async def stream_response(self, state: NewAgentState) -> AsyncGenerator[str, None]:
        """Stream the response token by token for real-time chat."""
        prompt = "\n".join([m["parts"][0] if isinstance(m["parts"][0], str) else m["parts"][0]["text"] for m in state.messages])
        if state.question or state.summarize:
            text_context = "\n".join(state.multimodal_context.get("text", [])[:500]) if state.multimodal_context else ""
            image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in state.multimodal_context.get("images", [])[:5]]) if state.multimodal_context else ""
            task = "summarize" if state.summarize else "answer"
            prompt = (
                f"{task.capitalize()} the following: {state.question or 'this document'}\n"
                f"Text Context: {text_context}\n"
                f"Image Context: {image_context}\n"
                "Provide a response:"
            )
        
        async for token in self.llm_client.generate_stream(prompt, image_base64=None if not state.multimodal_context or not state.multimodal_context["images"] else state.multimodal_context["images"][0]):
            yield token

    def build_agent(self):
        builder = StateGraph(NewAgentState)
        builder.add_node("preprocess_document", self.preprocess_document)
        builder.add_node("process_message", self.process_message)
        builder.add_node("stream_response", self.stream_response)
        
        builder.add_edge(START, "preprocess_document")
        builder.add_conditional_edges(
            "preprocess_document",
            lambda x: "process_message" if x.messages or x.question or x.summarize else END,
            path_map={"process_message": "stream_response", "stream_response": END}
        )
        builder.add_edge("stream_response", END)
        
        return builder.compile()
