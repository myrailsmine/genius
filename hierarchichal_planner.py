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
from utils.memory_store import MemoryStore
from agent_registry import AgentRegistry
from agents.router_agent import RouterState
from agents.document_rag_agent import DocumentRAGState
from agents.document_parsing_agent import DocumentLayoutParsingState
from agents.new_agent import NewAgentState

class HierarchicalPlanner:
    """
    A utility class for generating and optimizing hierarchical task plans with ReAct reasoning,
    supporting multimodal inputs, asynchronous processing, and team execution.
    """
    def __init__(self, llm_client: LLMClient = None, communicator: AgentCommunicator = None, toolkit: ToolKit = None, memory_store: MemoryStore = None):
        self.llm_client = llm_client or LLMClient()
        self.communicator = communicator or AgentCommunicator()
        self.toolkit = toolkit or ToolKit()
        self.memory_store = memory_store or MemoryStore()
        self.reasoning_cache = {}  # Cache for intermediate reasoning steps

    async def reasoning_loop(self, query: str, multimodal_context: Optional[dict] = None, max_steps: int = 5) -> List[Dict]:
        """
        Implement ReAct-style reasoning: iterate between reasoning and acting to refine the plan.

        Args:
            query (str): The user query to reason about.
            multimodal_context (Optional[dict]): Multimodal input for context.
            max_steps (int): Maximum reasoning steps to prevent infinite loops.

        Returns:
            List[Dict]: List of reasoning steps with actions and observations.
        """
        steps = []
        current_state = query
        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        for step in range(max_steps):
            # Step 1: Reason - Generate a reasoning step
            prompt = (
                f"Current state: {current_state}\n"
                f"Text Context: {text_context}\n"
                f"Image Context: {image_context}\n"
                "Reason about the next step to solve this query. Identify the next action (e.g., retrieve data, ask clarifying question, delegate to agent). "
                "Return a JSON object with 'reasoning', 'action', and 'next_state'."
            )
            reasoning_response = await self.llm_client.generate(prompt, max_length=MAX_LENGTH, temperature=0.7, top_p=TOP_P, stop=STOP_SEQUENCES)
            try:
                reasoning_step = json.loads(reasoning_response)
                if not all(key in reasoning_step for key in ["reasoning", "action", "next_state"]):
                    raise ValueError("Invalid reasoning step format")
            except Exception as e:
                await AsyncLogger.error(f"Failed to parse reasoning step: {e}")
                reasoning_step = {"reasoning": "Failed to reason", "action": "abort", "next_state": current_state}

            steps.append(reasoning_step)
            current_state = reasoning_step["next_state"]

            # Step 2: Act - Execute the action
            action = reasoning_step["action"]
            if "retrieve" in action.lower():
                documents = await self.toolkit.search_web(current_state) if "web" in action.lower() else []
                observation = f"Retrieved documents: {documents[:500]}" if documents else "No documents found"
            elif "delegate" in action.lower():
                observation = f"Delegating to agent: {current_state}"
            else:
                observation = "No action taken"

            # Update state with observation
            steps[-1]["observation"] = observation
            current_state = f"{current_state}\nObservation: {observation}"

            # Step 3: Check termination
            if "complete" in reasoning_step["action"].lower() or step == max_steps - 1:
                break

        self.reasoning_cache[query] = steps
        await self.memory_store.store_memory(f"reasoning_{query}", {"steps": steps, "query": query, "timestamp": time.time()})
        return steps

    async def plan_task(self, query: str, multimodal_context: Optional[dict] = None) -> List[Dict]:
        """
        Generate a hierarchical task plan with ReAct reasoning steps.

        Args:
            query (str): The user query or task to break down.
            multimodal_context (Optional[dict]): Dictionary with 'text' and 'images' (base64 strings) for multimodal input.

        Returns:
            List[Dict]: List of tasks with 'agent', 'subtask', 'priority', and 'reasoning_steps'.
        """
        await AsyncLogger.info(f"Planning task for query: {query}")
        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        # Check memory for past reasoning
        past_reasoning = await self.memory_store.retrieve_memory(f"reasoning_{query}")
        if past_reasoning:
            await AsyncLogger.info(f"Using cached reasoning for query: {query}")
            reasoning_steps = past_reasoning["steps"]
        else:
            reasoning_steps = await self.reasoning_loop(query, multimodal_context)

        # Only generate a plan if the query is complex or long enough
        if len(query.split()) <= PLANNING_THRESHOLD and not any(keyword in query.lower() for keyword in ["plan", "break down", "steps", "multimodal"]):
            await AsyncLogger.info(f"Query too simple for planning: {query}")
            return [{"agent": "rag_agent", "subtask": query, "priority": 5, "reasoning_steps": reasoning_steps}]

        prompt = (
            f"Break down this query into sub-tasks for an agentic RAG framework: {query}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            f"Reasoning Steps: {json.dumps(reasoning_steps)}\n"
            "Return a JSON list of tasks, each with 'agent' (agent name), 'subtask' (description), 'priority' (0-10), and 'reasoning_steps'. "
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
                task["reasoning_steps"] = reasoning_steps
            await AsyncLogger.info(f"Generated plan in {duration:.2f} seconds: {plan}")
            return plan
        except Exception as e:
            await AsyncLogger.error(f"Failed to parse task plan: {e}")
            return [{"agent": "rag_agent", "subtask": query, "priority": 5, "reasoning_steps": reasoning_steps}]

    async def optimize_plan(self, plan: List[Dict], multimodal_context: Optional[dict] = None) -> List[Dict]:
        """
        Optimize a hierarchical task plan for efficiency, preserving reasoning steps.
        """
        await AsyncLogger.info(f"Optimizing plan: {plan}")
        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        prompt = (
            f"Optimize this task plan for efficiency: {json.dumps(plan)}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Return an optimized JSON list of tasks, each with 'agent', 'subtask', 'priority' (0-10), and 'reasoning_steps', prioritizing efficiency and relevance."
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
            for task in optimized_plan:
                if not isinstance(task["priority"], int) or not (0 <= task["priority"] <= 10):
                    task["priority"] = 5
                task["reasoning_steps"] = plan[0]["reasoning_steps"] if plan else []
            await AsyncLogger.info(f"Optimized plan in {duration:.2f} seconds: {optimized_plan}")
            return optimized_plan
        except Exception as e:
            await AsyncLogger.error(f"Failed to parse optimized plan: {e}")
            return plan

    async def stream_plan(self, query: str, multimodal_context: Optional[dict] = None) -> AsyncGenerator[str, None]:
        text_context = "\n".join(multimodal_context.get("text", [])[:500]) if multimodal_context and "text" in multimodal_context else ""
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in multimodal_context.get("images", [])[:5]]) if multimodal_context and "images" in multimodal_context else ""

        reasoning_steps = await self.reasoning_loop(query, multimodal_context)
        prompt = (
            f"Break down this query into sub-tasks for an agentic RAG framework: {query}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            f"Reasoning Steps: {json.dumps(reasoning_steps)}\n"
            "Stream a JSON list of tasks, each with 'agent', 'subtask', 'priority' (0-10), and 'reasoning_steps'."
        )
        async for token in self.llm_client.generate_stream(prompt,
                                                         image_base64=None if not multimodal_context or not multimodal_context["images"] else multimodal_context["images"][0]):
            yield token

    async def validate_plan(self, plan: List[Dict]) -> bool:
        if not isinstance(plan, list):
            await AsyncLogger.warning("Plan must be a list")
            return False
        valid_agents = {"document_qa_agent", "term_sheet_agent", "db_assistant_agent", "rag_agent",
                       "document_multi_tool_agent", "parsing_agent", "router_agent", "new_agent"}
        for task in plan:
            if not isinstance(task, dict) or not all(key in task for key in ["agent", "subtask", "priority", "reasoning_steps"]):
                await AsyncLogger.warning(f"Invalid task format: {task}")
                return False
            if task["agent"] not in valid_agents:
                await AsyncLogger.warning(f"Invalid agent in plan: {task['agent']}")
                return False
            if not isinstance(task["priority"], int) or not (0 <= task["priority"] <= 10):
                await AsyncLogger.warning(f"Invalid priority in plan: {task['priority']}")
                return False
        return True

    async def execute_plan(self, plan: List[Dict], team: AgentTeam, parallel_tasks: bool = False) -> List[str]:
        if not await self.validate_plan(plan):
            await AsyncLogger.error("Invalid plan, execution aborted")
            return ["Plan validation failed"]

        await AsyncLogger.info(f"Executing plan with {len(plan)} tasks (parallel: {parallel_tasks})")
        results = []
        if parallel_tasks:
            tasks = [self._execute_task(task, team) for task in sorted(plan, key=lambda x: x["priority"], reverse=True)]
            results = await asyncio.gather(*tasks)
        else:
            for task in sorted(plan, key=lambda x: x["priority"], reverse=True):
                result = await self._execute_task(task, team)
                results.append(result)
        return results

    async def _execute_task(self, task: Dict, team: AgentTeam) -> str:
        agent_instance = AgentRegistry.get_agent(task["agent"].lower().replace("_agent", ""))()
        state_class = next((s for s in [RouterState, DocumentRAGState, DocumentLayoutParsingState, NewAgentState]
                          if s.__name__.lower() in task["agent"].lower()), RouterState)
        state = state_class(subtask=task["subtask"], query=task["subtask"] if "query" in state_class.__init__.__code__.co_varnames else None)
        result = await agent_instance.graph.ainvoke(state)
        task_result = result["result"]["response"] if "response" in result["result"] else "No response"
        await self.communicator.send_message("hierarchical_planner", task["agent"], f"Completed sub-task: {task['subtask']}", {"result": result["result"]})
        await self.memory_store.store_memory(f"task_{task['subtask']}", {"result": task_result, "task": task, "timestamp": time.time()})
        return task_result

    async def execute_plan_stream(self, plan: List[Dict], team: AgentTeam, parallel_tasks: bool = False) -> AsyncGenerator[str, None]:
        if not await self.validate_plan(plan):
            yield "Invalid plan, execution aborted\n"
            return

        await AsyncLogger.info(f"Streaming execution of plan with {len(plan)} tasks (parallel: {parallel_tasks})")
        if parallel_tasks:
            tasks = [(task, team) for task in sorted(plan, key=lambda x: x["priority"], reverse=True)]
            async for token in self._execute_parallel_stream(tasks):
                yield token
        else:
            for task in sorted(plan, key=lambda x: x["priority"], reverse=True):
                yield f"Executing sub-task: {task['subtask']} with {task['agent']} (Priority: {task['priority']})\n"
                async for token in self._execute_task_stream(task, team):
                    yield token
                yield f"Sub-task completed: {task['subtask']}\n"
        yield "Plan execution complete\n"

    async def _execute_task_stream(self, task: Dict, team: AgentTeam) -> AsyncGenerator[str, None]:
        agent_instance = AgentRegistry.get_agent(task["agent"].lower().replace("_agent", ""))()
        state_class = next((s for s in [RouterState, DocumentRAGState, DocumentLayoutParsingState, NewAgentState]
                          if s.__name__.lower() in task["agent"].lower()), RouterState)
        state = state_class(subtask=task["subtask"], query=task["subtask"] if "query" in state_class.__init__.__code__.co_varnames else None)
        async for token in agent_instance.graph.ainvoke(state, start_node="stream_response"):
            yield token
        result = await agent_instance.graph.ainvoke(state)
        await self.communicator.send_message("hierarchical_planner", task["agent"], f"Completed sub-task: {task['subtask']}", {"result": result["result"]})
        await self.memory_store.store_memory(f"task_{task['subtask']}", {"result": result["result"]["response"], "task": task, "timestamp": time.time()})

    async def _execute_parallel_stream(self, tasks: List[tuple]) -> AsyncGenerator[str, None]:
        for task, team in tasks:
            yield f"Executing sub-task: {task['subtask']} with {task['agent']} (Priority: {task['priority']})\n"
        futures = [self._execute_task_stream(task, team) for task, team in tasks]
        for future in asyncio.as_completed(futures):
            async for token in future:
                yield token
            yield "Sub-task completed\n"
