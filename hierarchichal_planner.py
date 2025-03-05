from typing import List, Dict, Optional, AsyncGenerator
from utils.llm_client import LLMClient
from utils.config import MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES, PLANNING_THRESHOLD
from utils.logger import logger, AsyncLogger
import asyncio
import json
import time
from langchain_core.documents import Document

class HierarchicalPlanner:
    """
    A utility class for generating and optimizing hierarchical task plans for agentic workflows,
    supporting multimodal inputs and asynchronous processing.
    """
    def __init__(self):
        self.llm_client = LLMClient()

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
        logger.info(f"Planning task for query: {query}")
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
            "Agents can be: document_qa_agent, term_sheet_agent, db_assistant_agent, rag_agent, document_multi_tool_agent. "
            "Ensure priorities are integers between 0 and 10, with higher numbers indicating higher urgency."
        )
        start_time = time.time()
        response = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0], max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP_SEQUENCES)
        duration = time.time() - start_time

        try:
            plan = json.loads(response)
            if not isinstance(plan, list) or not all("agent" in task and "subtask" in task and "priority" in task for task in plan):
                raise ValueError("Invalid plan format: must be a list of tasks with 'agent', 'subtask', and 'priority'")
            # Validate priorities are integers between 0 and 10
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
        logger.info(f"Optimizing plan: {plan}")
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
        response = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0], max_length=MAX_LENGTH, temperature=0.1, top_p=0.9, stop=STOP_SEQUENCES)
        duration = time.time() - start_time

        try:
            optimized_plan = json.loads(response)
            if not isinstance(optimized_plan, list) or not all("agent" in task and "subtask" in task and "priority" in task for task in optimized_plan):
                raise ValueError("Invalid optimized plan format")
            # Validate priorities are integers between 0 and 10
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
        async for token in self.llm_client.generate_stream(prompt, image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0]):
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
            logger.warning("Plan must be a list")
            return False
        valid_agents = {"document_qa_agent", "term_sheet_agent", "db_assistant_agent", "rag_agent", "document_multi_tool_agent"}
        for task in plan:
            if not isinstance(task, dict) or not all(key in task for key in ["agent", "subtask", "priority"]):
                logger.warning(f"Invalid task format: {task}")
                return False
            if task["agent"] not in valid_agents:
                logger.warning(f"Invalid agent in plan: {task['agent']}")
                return False
            if not isinstance(task["priority"], int) or not (0 <= task["priority"] <= 10):
                logger.warning(f"Invalid priority in plan: {task['priority']}")
                return False
        return True
