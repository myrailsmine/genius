from typing import Optional, Dict, AsyncGenerator
from utils.llm_client import LLMClient
from utils.config import MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES, REFLECTION_ENABLED, PLANNING_THRESHOLD
from utils.logger import AsyncLogger, logger
import asyncio
import json
import time
from langchain_core.documents import Document
from utils.agent_communication import AgentMessage, AgentCommunicator
from agent_registry import AgentRegistry

class ReflectiveAgent:
    """
    A utility class for self-reflection and improvement of agent performance,
    supporting multimodal inputs, asynchronous processing, and team-based learning.
    """
    def __init__(self, llm_client: LLMClient = None, communicator: AgentCommunicator = None):
        """
        Initialize the ReflectiveAgent with an LLM client and optional communicator for team interaction.

        Args:
            llm_client (LLMClient, optional): The LLM client instance for generating reflection responses.
            communicator (AgentCommunicator, optional): Communicator for sending improvement messages to other agents.
        """
        self.llm_client = llm_client or LLMClient()
        self.communicator = communicator or AgentCommunicator()
        self.enabled = REFLECTION_ENABLED
        self.improvements = {}  # Store improvements by task ID

    async def reflect(self, query: str, response: str, feedback: Optional[str] = None, multimodal_context: Optional[dict] = None) -> Dict:
        """
        Reflect on an interaction to identify errors, suggest improvements, and propose a refined strategy,
        optionally using multimodal context (text + images).

        Args:
            query (str): The original user query or task.
            response (str): The agent’s response to evaluate.
            feedback (Optional[str]): Optional user feedback on the response.
            multimodal_context (Optional[dict]): Dictionary with 'text' and 'images' (base64 strings) for multimodal input.

        Returns:
            Dict: Reflection result with 'errors', 'improvements', and 'strategy'.

        Raises:
            ValueError: If reflection generation fails or the response is malformed.
        """
        if not self.enabled:
            logger.warning("Reflection is disabled in config")
            return {"errors": "None", "improvements": "None", "strategy": "Maintain current approach"}

        logger.info(f"Reflecting on query: {query}, response: {response}, feedback: {feedback}")
        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        prompt = (
            f"Query: {query}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback or 'None'}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Reflect on this interaction: Identify errors, suggest improvements, and propose a refined strategy. "
            "Return a JSON object with 'errors', 'improvements', and 'strategy'."
        )
        start_time = time.time()
        response_text = await asyncio.to_thread(self.llm_client.generate, prompt, 
                                              image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0], 
                                              max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP_SEQUENCES)
        duration = time.time() - start_time

        try:
            reflection = json.loads(response_text)
            if not isinstance(reflection, dict) or not all(key in reflection for key in ["errors", "improvements", "strategy"]):
                raise ValueError("Invalid reflection format: must include 'errors', 'improvements', and 'strategy'")
            task_id = f"reflection_{int(time.time())}"
            self.improvements[task_id] = reflection
            await AsyncLogger.info(f"Reflection completed in {duration:.2f} seconds: {reflection}")
            await self.communicator.send_message("reflective_agent", "router_agent", f"New reflection for task {task_id}: {reflection}", {"id": task_id})
            return reflection
        except Exception as e:
            await AsyncLogger.error(f"Failed to parse reflection: {e}")
            return {"errors": "None", "improvements": "None", "strategy": "Maintain current approach"}

    async def stream_reflection(self, query: str, response: str, feedback: Optional[str] = None, multimodal_context: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """
        Stream a reflection token by token for real-time display.

        Args:
            query (str): The original user query or task.
            response (str): The agent’s response to evaluate.
            feedback (Optional[str]): Optional user feedback on the response.
            multimodal_context (Optional[dict]): Dictionary with 'text' and 'images' (base64 strings) for multimodal input.

        Yields:
            str: Tokens of the reflection as they are generated.
        """
        if not self.enabled:
            yield json.dumps({"errors": "None", "improvements": "None", "strategy": "Maintain current approach"})
            return

        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        prompt = (
            f"Query: {query}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback or 'None'}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Reflect on this interaction: Stream a JSON object with 'errors', 'improvements', and 'strategy'."
        )
        async for token in self.llm_client.generate_stream(prompt, 
                                                        image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0]):
            yield token
            # Aggregate tokens into a valid JSON object for streaming
            if "}" in token:
                try:
                    partial_json = "".join([t for t in self._stream_buffer if t])
                    reflection = json.loads(partial_json)
                    if all(key in reflection for key in ["errors", "improvements", "strategy"]):
                        self._stream_buffer = []  # Reset buffer on valid JSON
                        task_id = f"reflection_{int(time.time())}"
                        self.improvements[task_id] = reflection
                        await self.communicator.send_message("reflective_agent", "router_agent", f"New reflection for task {task_id}: {reflection}", {"id": task_id})
                except json.JSONDecodeError:
                    continue
        yield ""  # Ensure generator completes

    async def apply_improvements(self, reflection: Dict, agent_strategy: str, agent: object = None) -> str:
        """
        Apply reflection improvements to refine an agent's strategy and notify the agent.

        Args:
            reflection (Dict): Reflection result with 'errors', 'improvements', and 'strategy'.
            agent_strategy (str): Current strategy of the agent.
            agent (object, optional): The agent instance to apply improvements to directly.

        Returns:
            str: Refined strategy based on reflection.

        Raises:
            ValueError: If the strategy refinement fails or the reflection is invalid.
        """
        if not self.enabled or not reflection["improvements"]:
            return agent_strategy

        await AsyncLogger.info(f"Applying improvements to strategy: {agent_strategy}")
        prompt = (
            f"Current agent strategy: {agent_strategy}\n"
            f"Reflection: {json.dumps(reflection)}\n"
            "Refine this strategy based on the reflection's improvements. Return the refined strategy as a string."
        )
        start_time = time.time()
        response = await asyncio.to_thread(self.llm_client.generate, prompt, max_length=MAX_LENGTH, temperature=0.5, top_p=0.9, stop=STOP_SEQUENCES)
        duration = time.time() - start_time

        try:
            refined_strategy = response.strip()
            if not isinstance(refined_strategy, str) or not refined_strategy:
                raise ValueError("Invalid refined strategy format")
            await AsyncLogger.info(f"Refined strategy in {duration:.2f} seconds: {refined_strategy}")

            # Apply improvement directly to the agent if provided
            if agent and hasattr(agent, "update_strategy"):
                await agent.update_strategy(reflection["improvements"])
                await self.communicator.send_message("reflective_agent", agent.__class__.__name__.lower().replace("_agent", ""), f"Applied improvement: {reflection['improvements']}", {"id": list(self.improvements.keys())[-1]})

            return refined_strategy
        except Exception as e:
            await AsyncLogger.error(f"Failed to refine strategy: {e}")
            return agent_strategy

    # Internal buffer for streaming JSON aggregation
    _stream_buffer = []

    async def improve_agent_team(self, team: List[object], task_id: str):
        """
        Improve a team of agents based on stored reflections for a specific task.

        Args:
            team (List[object]): List of agent instances to improve.
            task_id (str): The ID of the task to apply improvements for.
        """
        if task_id in self.improvements:
            reflection = self.improvements[task_id]
            for agent in team:
                if hasattr(agent, "update_strategy"):
                    await agent.update_strategy(reflection["improvements"])
                    await AsyncLogger.info(f"Improved {agent.__class__.__name__} with: {reflection['improvements']}")
                    await self.communicator.send_message("reflective_agent", agent.__class__.__name__.lower().replace("_agent", ""), f"Team improvement applied: {reflection['improvements']}", {"id": task_id})
