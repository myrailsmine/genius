from langgraph.graph import StateGraph, END
from utils.image_utils import base64_to_pil_image
from utils.logger import AsyncLogger
from pypdf import PdfReader
from pdf2image import convert_from_path
import io
import base64
from utils.agent_communication import AgentMessage
from agent_registry import AgentRegistry
import asyncio

class DocumentLayoutParsingState:
    def __init__(self, document_path=None, messages=None, multimodal_context=None):
        self.document_path = document_path
        self.pages_as_base64_jpeg_images = []
        self.pages_as_text = []
        self.documents = []
        self.messages = messages or []
        self.multimodal_context = multimodal_context or {"text": [], "images": []}
        self.result = {}

class DocumentParsingAgent:
    def __init__(self):
        self.graph = StateGraph(DocumentLayoutParsingState)
        self._build_graph()

    def _build_graph(self):
        self.graph.add_node("parse_document", self.parse_document)
        self.graph.set_entry_point("parse_document")
        self.graph.add_edge("parse_document", END)

    async def parse_document(self, state: DocumentLayoutParsingState):
        await AsyncLogger.info(f"Parsing document: {state.document_path}")
        reader = PdfReader(state.document_path)
        images = convert_from_path(state.document_path)
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()
            state.pages_as_text.append(text)
            if images:
                image = images[page_num]
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                state.pages_as_base64_jpeg_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
        state.documents = [{"page_content": t, "metadata": {"element_type": "Text-block"}} for t in state.pages_as_text]
        state.documents.extend([{"page_content": "", "metadata": {"element_type": "Image", "image_base64": img}} for img in state.pages_as_base64_jpeg_images])
        if state.messages:
            for msg in state.messages:
                if msg.sender != self.__class__.__name__:
                    await self.process_message(msg)
        return state

    async def process_message(self, message: AgentMessage):
        await AsyncLogger.info(f"Received message from {message.sender}: {message.content}")
        if "reparse" in message.content.lower():
            return await self.parse_document({"document_path": message.task.get("document_path")})
        return "Parsing complete"

    async def update_strategy(self, improvement: str):
        await AsyncLogger.info(f"Updating strategy with: {improvement}")
        # Hypothetical logic to adjust parsing (e.g., image quality)
        if "improve image quality" in improvement.lower():
            pass  # Implement quality adjustment logic

    async def graph(self, state: DocumentLayoutParsingState, start_node=None):
        async for event in self.graph.stream(state, start_node=start_node):
            yield event
