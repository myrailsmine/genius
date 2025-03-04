from typing import Dict, Type, Callable, Optional, List
from pydantic import BaseModel
import asyncio
from document_ai_agents.logger import logger
from utils.config import LOG_LEVEL

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class AgentMetadata(BaseModel):
    """
    Metadata for an agent, describing its capabilities and supported document types.
    """
    name: str
    capabilities: List[str]  # e.g., ["qa", "summarization", "multimodal", "tool_use"]
    supported_doc_types: List[str]  # e.g., ["generic", "term_sheet", "research_paper", "database"]

class BaseAgentState(BaseModel):
    """
    Base state model for agents, extensible by specific agents.
    """
    messages: List[dict] = []  # For chat-based agents
    document_path: Optional[str] = None
    question: Optional[str] = None
    summarize: bool = False
    documents: List[Document] = []  # For document-based agents
    multimodal_context: Optional[dict] = None  # For multimodal support (text + images)
    database_connection: Optional[str] = None  # For database agents
    knowledge_base_id: Optional[str] = None  # For knowledge base integration

class AgentRegistry:
    """
    Registry for managing agent classes and their metadata, supporting asynchronous operations.
    """
    def __init__(self):
        self.agents: Dict[str, tuple[Type, AgentMetadata]] = {}
        self.tools: Dict[str, Callable] = {}  # For tool registration

    def register(self, name: str, capabilities: List[str], supported_doc_types: List[str]) -> Callable:
        """
        Decorator to register an agent class with its metadata.
        
        Args:
            name (str): Unique name of the agent.
            capabilities (List[str]): List of capabilities (e.g., "qa", "summarization").
            supported_doc_types (List[str]): List of supported document types (e.g., "generic", "term_sheet").
        
        Returns:
            Callable: Decorator function for the agent class.
        """
        def decorator(cls):
            metadata = AgentMetadata(name=name, capabilities=capabilities, supported_doc_types=supported_doc_types)
            self.agents[name] = (cls, metadata)
            logger.info(f"Registered agent: {name} with capabilities {capabilities} and supported doc types {supported_doc_types}")
            return cls
        return decorator

    async def get_agent(self, name: str) -> Optional[Type]:
        """
        Asynchronously retrieve an agent class by name.
        
        Args:
            name (str): Name of the agent to retrieve.
        
        Returns:
            Optional[Type]: Agent class if found, None otherwise.
        """
        agent_class, _ = self.agents.get(name, (None, None))
        if not agent_class:
            logger.warning(f"Agent not found: {name}")
        return agent_class

    def get_agent_metadata(self, name: str) -> Optional[AgentMetadata]:
        """
        Retrieve metadata for an agent by name.
        
        Args:
            name (str): Name of the agent.
        
        Returns:
            Optional[AgentMetadata]: Agent metadata if found, None otherwise.
        """
        _, metadata = self.agents.get(name, (None, None))
        if not metadata:
            logger.warning(f"Metadata not found for agent: {name}")
        return metadata

    def list_agents(self) -> Dict[str, AgentMetadata]:
        """
        List all registered agents and their metadata.
        
        Returns:
            Dict[str, AgentMetadata]: Dictionary of agent names and their metadata.
        """
        return {name: meta for name, (_, meta) in self.agents.items()}

    def register_tool(self, name: str, tool: Callable) -> None:
        """
        Register a tool for use by agents.
        
        Args:
            name (str): Unique name of the tool.
            tool (Callable): Tool function to register.
        """
        self.tools[name] = tool
        logger.info(f"Registered tool: {name}")

    async def get_tool(self, name: str) -> Optional[Callable]:
        """
        Asynchronously retrieve a tool by name.
        
        Args:
            name (str): Name of the tool to retrieve.
        
        Returns:
            Optional[Callable]: Tool function if found, None otherwise.
        """
        tool = self.tools.get(name)
        if not tool:
            logger.warning(f"Tool not found: {name}")
        return tool

agent_registry = AgentRegistry()

# Example usage for testing
if __name__ == "__main__":
    # Example agent registration
    @agent_registry.register("example_agent", ["qa", "multimodal"], ["generic"])
    class ExampleAgent:
        def __init__(self):
            self.llm_client = LLMClient()
        
        async def process(self, state: BaseAgentState):
            response = await self.llm_client.generate(state.question or "Provide a summary")
            return {"messages": state.messages + [{"role": "system", "parts": [response]}]
