from typing import List, Dict, Optional, AsyncGenerator
from utils.llm_client import LLMClient
from utils.config import MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES, PLANNING_THRESHOLD
from utils.logger import AsyncLogger, logger
import asyncio
import json
import time
from langchain_core.documents import Document
from utils.agent_team import AgentTeam
from utils.agent_communication import AgentCommunicator
from utils.tools import ToolKit
from agent_registry import AgentRegistry
from agents.router_agent import RouterState
from agents.document_rag_agent import DocumentRAGState
from agents.document_parsing_agent import DocumentLayoutParsingState
from agents.new_agent import NewAgentState

class HierarchicalPlanner:
    """
    A utility class for generating and optimizing hierarchical task plans for agentic workflows,
    supporting multimodal inputs, asynchronous processing, and team execution.
    """
    def __init__(self, llm_client: LLMClient = None, communicator: AgentCommunicator = None, toolkit: ToolKit = None):
        """
        Initialize the HierarchicalPlanner with an LLM client, communicator, and toolkit.

        Args:
            llm_client (LLMClient, optional): The LLM client instance for generating plans.
            communicator (AgentCommunicator, optional): Communicator for coordinating with agents.
            toolkit (ToolKit, optional): Toolkit for accessing external resources (e.g., web search).
        """
        self.llm_client = llm_client or LLMClient()
        self.communicator = communicator or AgentCommunicator()
        self.toolkit = toolkit or ToolKit()

    async def plan_task(self, query: str, multimodal_context: Optional[dict] = None) -> List[Dict]:
        """
        Generate a hierarchical task plan for a given query, optionally using multimodal context (text + images).

        Args:
            query (str): The user query or task to break down.
            multimodal_context (Optional[dict]): Dictionary with 'text' and 'images' (base64 strings) for multimodal input.

        Returns:
            List[Dict]: List of tasks, each with 'agent', 'subtask', and 'priority' (0-10).

        Raises:
            ValueError: If the plan generation fails or the response is malformed.
        """
        await AsyncLogger.info(f"Planning task for query: {query}")
        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        # Only generate a plan if the query is complex or long enough
        if len(query.split()) <= PLANNING_THRESHOLD and not any(keyword in query.lower() for keyword in ["plan", "break down", "steps", "multimodal"]):
            await AsyncLogger.info(f"Query too simple for planning: {query}")
            return [{"agent": "rag_agent", "subtask": query, "priority": 5}]

        prompt = (
            f"Break down this query into sub-tasks for an agentic RAG framework: {query}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Return a JSON list of tasks, each with 'agent' (agent name), 'subtask' (description), and 'priority' (0-10). "
            "Agents can be: document_qa_agent, term_sheet_agent, db_assistant_agent, rag_agent, document_multi_tool_agent, parsing_agent, router_agent, new_agent. "
            "Ensure priorities are integers between 0 and 10, with higher numbers indicating higher urgency."
        )
        start_time = time.time()
        response = await asyncio.to_thread(self.llm_client.generate, prompt, 
                                        image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0], 
                                        max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP_SEQUENCES)
        duration = time.time() - start_time

        try:
            plan = json.loads(response)
            if not isinstance(plan, list) or not all("agent" in task and "subtask" in task and "priority" in task for task in plan):
                raise ValueError("Invalid plan format: must be a list of tasks with 'agent', 'subtask', and 'priority'")
            # Validate and normalize priorities
            for task in plan:
                if not isinstance(task["priority"], int) or not (0 <= task["priority"] <= 10):
                    task["priority"] = 5  # Default priority if invalid
            await AsyncLogger.info(f"Generated plan in {duration:.2f} seconds: {plan}")
            return plan
        except Exception as e:
            await AsyncLogger.error(f"Failed to parse task plan: {e}")
            return [{"agent": "rag_agent", "subtask": query, "priority": 5}]

    async def optimize_plan(self, plan: List[Dict], multimodal_context: Optional[dict] = None) -> List[Dict]:
        """
        Optimize a hierarchical task plan for efficiency, optionally using multimodal context.

        Args:
            plan (List[Dict]): Initial plan with tasks containing 'agent', 'subtask', and 'priority'.
            multimodal_context (Optional[dict]): Dictionary with 'text' and 'images' (base64 strings) for multimodal input.

        Returns:
            List[Dict]: Optimized plan with refined tasks and priorities.

        Raises:
            ValueError: If the optimization fails or the response is malformed.
        """
        await AsyncLogger.info(f"Optimizing plan: {plan}")
        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        prompt = (
            f"Optimize this task plan for efficiency: {json.dumps(plan)}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Return an optimized JSON list of tasks, each with 'agent', 'subtask', and 'priority' (0-10), prioritizing efficiency and relevance. "
            "Ensure priorities are integers between 0 and 10, with higher numbers indicating higher urgency."
        )
        start_time = time.time()
        response = await asyncio.to_thread(self.llm_client.generate, prompt, 
                                        image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0], 
                                        max_length=MAX_LENGTH, temperature=0.1, top_p=0.9, stop=STOP_SEQUENCES)
        duration = time.time() - start_time

        try:
            optimized_plan = json.loads(response)
            if not isinstance(optimized_plan, list) or not all("agent" in task and "subtask" in task and "priority" in task for task in optimized_plan):
                raise ValueError("Invalid optimized plan format")
            # Validate and normalize priorities
            for task in optimized_plan:
                if not isinstance(task["priority"], int) or not (0 <= task["priority"] <= 10):
                    task["priority"] = 5  # Default priority if invalid
            await AsyncLogger.info(f"Optimized plan in {duration:.2f} seconds: {optimized_plan}")
            return optimized_plan
        except Exception as e:
            await AsyncLogger.error(f"Failed to parse optimized plan: {e}")
            return plan

    async def stream_plan(self, query: str, multimodal_context: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """
        Stream a hierarchical task plan token by token for real-time display.

        Args:
            query (str): The user query or task to break down.
            multimodal_context (Optional[dict]): Dictionary with 'text' and 'images' (base64 strings) for multimodal input.

        Yields:
            str: Tokens of the plan as they are generated.
        """
        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        prompt = (
            f"Break down this query into sub-tasks for an agentic RAG framework: {query}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Stream a JSON list of tasks, each with 'agent', 'subtask', and 'priority' (0-10)."
        )
        async for token in self.llm_client.generate_stream(prompt, 
                                                        image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0]):
            yield token

    async def validate_plan(self, plan: List[Dict]) -> bool:
        """
        Validate a task plan to ensure it meets the framework's requirements.

        Args:
            plan (List[Dict]): Plan to validate, containing tasks with 'agent', 'subtask', and 'priority'.

        Returns:
            bool: True if the plan is valid, False otherwise.
        """
        if not isinstance(plan, list):
            await AsyncLogger.warning("Plan must be a list")
            return False
        valid_agents = {"document_qa_agent", "term_sheet_agent", "db_assistant_agent", "rag_agent", 
                       "document_multi_tool_agent", "parsing_agent", "router_agent", "new_agent"}
        for task in plan:
            if not isinstance(task, dict) or not all(key in task for key in ["agent", "subtask", "priority"]):
                await AsyncLogger.warning(f"Invalid task format: {task}")
                return False
            if task["agent"] not in valid_agents:
                await AsyncLogger.warning(f"Invalid agent in plan: {task['agent']}")
                return False
            if not isinstance(task["priority"], int) or not (0 <= task["priority"] <= 10):
                await AsyncLogger.warning(f"Invalid priority in plan: {task['priority']}")
                return False
        return True

    async def execute_plan(self, plan: List[Dict], team: AgentTeam) -> List[str]:
        """
        Execute a hierarchical task plan using a team of agents.

        Args:
            plan (List[Dict]): List of tasks with 'agent', 'subtask', and 'priority'.
            team (AgentTeam): The team of agents to execute the plan.

        Returns:
            List[str]: List of results from each sub-task execution.
        """
        if not await self.validate_plan(plan):
            await AsyncLogger.error("Invalid plan, execution aborted")
            return ["Plan validation failed"]

        await AsyncLogger.info(f"Executing plan with {len(plan)} tasks")
        results = []
        for task in sorted(plan, key=lambda x: x["priority"], reverse=True):  # Execute by priority
            agent_instance = AgentRegistry.get_agent(task["agent"].lower().replace("_agent", ""))()
            state_class = next((s for s in [RouterState, DocumentRAGState, DocumentLayoutParsingState, NewAgentState] 
                              if s.__name__.lower() in task["agent"].lower()), RouterState)
            state = state_class(subtask=task["subtask"], query=task["subtask"] if "query" in state_class.__init__.__code__.co_varnames else None)
            result = await agent_instance.graph.ainvoke(state)
            results.append(result["result"]["response"] if "response" in result["result"] else "No response")
            await self.communicator.send_message("hierarchical_planner", task["agent"], f"Completed sub-task: {task['subtask']}", {"result": result["result"]})
        return results

    async def execute_plan_stream(self, plan: List[Dict], team: AgentTeam) -> AsyncGenerator[str, None]:
        """
        Stream the execution of a hierarchical task plan token by token.

        Args:
            plan (List[Dict]): List of tasks with 'agent', 'subtask', and 'priority'.
            team (AgentTeam): The team of agents to execute the plan.

        Yields:
            str: Tokens describing the execution progress.
        """
        if not await self.validate_plan(plan):
            yield "Invalid plan, execution aborted\n"
            return

        await AsyncLogger.info(f"Streaming execution of plan with {len(plan)} tasks")
        for task in sorted(plan, key=lambda x: x["priority"], reverse=True):  # Execute by priority
            yield f"Executing sub-task: {task['subtask']} with {task['agent']} (Priority: {task['priority']})\n"
            agent_instance = AgentRegistry.get_agent(task["agent"].lower().replace("_agent", ""))()
            state_class = next((s for s in [RouterState, DocumentRAGState, DocumentLayoutParsingState, NewAgentState] 
                              if s.__name__.lower() in task["agent"].lower()), RouterState)
            state = state_class(subtask=task["subtask"], query=task["subtask"] if "query" in state_class.__init__.__code__.co_varnames else None)
            async for token in agent_instance.graph.ainvoke(state, start_node="stream_response"):
                yield token
            result = await agent_instance.graph.ainvoke(state)
            yield f"Sub-task completed: {task['subtask']}\n"
            await self.communicator.send_message("hierarchical_planner", task["agent"], f"Completed sub-task: {task['subtask']}", {"result": result["result"]})
        yield "Plan execution complete\n"
