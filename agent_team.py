from typing import List, Dict, Optional
from utils.llm_client import LLMClient
from utils.logger import AsyncLogger
from utils.agent_communication import AgentCommunicator
from utils.tools import ToolKit
from agent_registry import AgentRegistry
import asyncio

class LoadBalancer:
    """
    A utility class to balance load across agents based on their current workload.
    """
    def __init__(self, agents: List[object]):
        self.agents = {agent.__class__.__name__.lower().replace("_agent", ""): {"instance": agent, "workload": 0} for agent in agents}
        self.max_workload = 5  # Maximum concurrent tasks per agent

    async def assign_task(self, task: Dict) -> object:
        """
        Assign a task to the least busy agent capable of handling it.

        Args:
            task (Dict): Task with 'agent' specifying the desired agent type.

        Returns:
            object: The assigned agent instance.
        """
        agent_name = task["agent"].lower().replace("_agent", "")
        if agent_name not in self.agents:
            await AsyncLogger.error(f"No agent available for type: {agent_name}")
            raise ValueError(f"No agent available for type: {agent_name}")

        agent_info = self.agents[agent_name]
        if agent_info["workload"] >= self.max_workload:
            await AsyncLogger.warning(f"Agent {agent_name} is overloaded, waiting...")
            await asyncio.sleep(1)  # Simple backoff; replace with queue in production
        agent_info["workload"] += 1
        return agent_info["instance"]

    async def task_completed(self, agent_name: str):
        """
        Mark a task as completed, reducing the agent's workload.
        """
        agent_name = agent_name.lower().replace("_agent", "")
        if agent_name in self.agents:
            self.agents[agent_name]["workload"] = max(0, self.agents[agent_name]["workload"] - 1)

class AgentTeam:
    """
    A team of agents working together to execute tasks, with parallel execution and load balancing.
    """
    def __init__(self, agent_types: List[str], llm_client: LLMClient = None, communicator: AgentCommunicator = None, toolkit: ToolKit = None):
        self.llm_client = llm_client or LLMClient()
