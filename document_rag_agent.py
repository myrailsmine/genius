from langgraph.graph import StateGraph, END
from utils.llm_client import LLMClient
from utils.hybrid_retriever import MultimodalRetriever
from utils.config import RETRIEVAL_K
from utils.logger import AsyncLogger
from utils.agent_communication import AgentMessage
from agent_registry import AgentRegistry
import asyncio

class DocumentRAGState:
    def __init__(self, question=None, document_path=None, pages_as_base64_jpeg_images=None, pages_as_text=None, documents=None, summarize=False, messages=None, multimodal_context=None):
        self.question = question
        self.document_path = document_path
        self.pages_as_base64_jpeg_images = pages_as_base64_jpeg_images or []
        self.pages_as_text = pages_as_text or []
        self.documents = documents or []
        self.summarize = summarize
        self.messages = messages or []
        self.multimodal_context = multimodal_context or {"text": [], "images": []}
        self.result = {}

class DocumentRAGAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.retriever = MultimodalRetriever([])
        self.graph = StateGraph(DocumentRAGState)
        self._build_graph()

    def _build_graph(self):
        self.graph.add_node("retrieve", self.retrieve)
        self.graph.add_node("answer", self.answer)
        self.graph.add_node("stream_answer", self.stream_answer)
        self.graph.set_entry_point("retrieve")
        self.graph.add_edge("retrieve", "answer")
        self.graph.add_edge("answer", END)
        self.graph.add_edge("stream_answer", END)

    async def retrieve(self, state: DocumentRAGState):
        documents = await self.retriever.retrieve(state.question or "Summarize this document" if state.summarize else "")
        state.documents = documents
        if state.messages:
            for msg in state.messages:
                if msg.sender != self.__class__.__name__:
                    await self.process_message(msg)
        return state

    async def answer(self, state: DocumentRAGState):
        context = " ".join([doc.page_content for doc in state.documents])
        prompt = f"{state.question}\nContext: {context}" if state.question else f"Summarize: {context}"
        state.result["response"] = await self.llm_client.generate(prompt)
        return state

    async def stream_answer(self, state: DocumentRAGState):
        context = " ".join([doc.page_content for doc in state.documents])
        prompt = f"{state.question}\nContext: {context}" if state.question else f"Summarize: {context}"
        async for token in self.llm_client.generate_stream(prompt):
            yield token

    async def process_message(self, message: AgentMessage):
        await AsyncLogger.info(f"Received message from {message.sender}: {message.content}")
        if "update" in message.content.lower():
            await self.update_strategy(message.content)
        return await self.llm_client.generate(f"Processed: {message.content}")

    async def update_strategy(self, improvement: str):
        await AsyncLogger.info(f"Updating strategy with: {improvement}")
        # Hypothetical logic to adjust retrieval or prompts
        if "increase retrieval" in improvement.lower():
            self.retriever.k = min(self.retriever.k + 1, 10)

    async def graph(self, state: DocumentRAGState, start_node=None):
        async for event in self.graph.stream(state, start_node=start_node):
            yield event
