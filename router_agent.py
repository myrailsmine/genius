from langgraph.graph import StateGraph, END
from utils.llm_client import LLMClient
from utils.hybrid_retriever import MultimodalRetriever
from utils.config import LOG_LEVEL
from utils.logger import AsyncLogger
from utils.agent_communication import AgentMessage
from utils.tools import ToolKit
from agent_registry import AgentRegistry
import asyncio

class RouterState:
    def __init__(self, document_path=None, query=None, summarize=False, knowledge_base_id=None, database_connection=None, feedback=None, messages=None, multimodal_context=None):
        self.document_path = document_path
        self.query = query
        self.summarize = summarize
        self.knowledge_base_id = knowledge_base_id
        self.database_connection = database_connection
        self.feedback = feedback
        self.messages = messages or []
        self.multimodal_context = multimodal_context or {"text": [], "images": []}
        self.result = {}

class RouterAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.retriever = MultimodalRetriever([])
        self.toolkit = ToolKit()
        self.doc_labels = ["qa", "summarize", "feedback"]
        self.graph = StateGraph(RouterState)
        self._build_graph()

    def _build_graph(self):
        self.graph.add_node("determine_agent", self.determine_agent)
        self.graph.add_node("stream_response", self.stream_response)
        self.graph.set_entry_point("determine_agent")
        self.graph.add_conditional_edges("determine_agent", self.route)
        self.graph.add_edge("stream_response", END)

    async def determine_agent(self, state: RouterState):
        await AsyncLogger.info(f"Determining agent for query: {state.query}")
        content = state.query or state.feedback or ""
        label = await asyncio.to_thread(self.llm_client.classify, content[:512], self.doc_labels)
        state.result["intent"] = label
        if state.messages:
            for msg in state.messages:
                if msg.sender != self.__class__.__name__:
                    await self.process_message(msg)
        return state

    def route(self, state: RouterState):
        intent = state.result.get("intent", {}).get("qa", 1.0) > 0.5
        if state.summarize or "summarize" in state.result.get("intent", {}):
            return "rag_agent"
        elif state.feedback:
            return "reflective_agent"
        elif intent:
            return "rag_agent"
        return END

    async def stream_response(self, state: RouterState):
        agent = AgentRegistry.get_agent("rag_agent")()
        async for token in agent.graph.ainvoke(state, start_node="stream_answer"):
            yield token

    async def process_message(self, message: AgentMessage):
        await AsyncLogger.info(f"Received message from {message.sender}: {message.content}")
        if "web_search" in message.content.lower():
            result = await self.toolkit.search_web(message.task.get("query", ""))
            return await self.llm_client.generate(f"Web search result: {result}")
        return await self.llm_client.generate(message.content)

    async def update_strategy(self, improvement: str):
        await AsyncLogger.info(f"Updating strategy with: {improvement}")
        # Hypothetical logic to adjust routing logic
        if "increase retrieval" in improvement.lower():
            self.retriever.k = min(self.retriever.k + 1, 10)

    async def graph(self, state: RouterState, start_node=None):
        async for event in self.graph.stream(state, start_node=start_node):
            yield event
