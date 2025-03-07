from langgraph.graph import StateGraph, END
from utils.llm_client import LLMClient
from utils.logger import AsyncLogger
from utils.agent_communication import AgentMessage
from agent_registry import AgentRegistry
import asyncio

class NewAgentState:
    def __init__(self, message=None, document_path=None, messages=None, multimodal_context=None):
        self.message = message
        self.document_path = document_path
        self.messages = messages or []
        self.multimodal_context = multimodal_context or {"text": [], "images": []}
        self.result = {}

class NewAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.graph = StateGraph(NewAgentState)
        self._build_graph()

    def _build_graph(self):
        self.graph.add_node("process_message", self.process_message)
        self.graph.add_node("stream_response", self.stream_response)
        self.graph.set_entry_point("process_message")
        self.graph.add_edge("process_message", "stream_response")
        self.graph.add_edge("stream_response", END)

    async def process_message(self, state: NewAgentState):
        if state.messages:
            for msg in state.messages:
                if msg.sender != self.__class__.__name__:
                    await self.process_message(msg)
        state.result["processed"] = await self.llm_client.generate(state.message)
        return state

    async def stream_response(self, state: NewAgentState):
        async for token in self.llm_client.generate_stream(state.message):
            yield token

    async def update_strategy(self, improvement: str):
        await AsyncLogger.info(f"Updating strategy with: {improvement}")
        # Hypothetical logic to refine prompts
        if "increase creativity" in improvement.lower():
            self.llm_client.temperature = min(self.llm_client.temperature + 0.1, 1.0)

    async def graph(self, state: NewAgentState, start_node=None):
        async for event in self.graph.stream(state, start_node=start_node):
            yield event
