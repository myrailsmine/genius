from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator, Optional, List, Dict
from agents.router_agent import RouterState, RouterAgent
from agents.document_rag_agent import DocumentRAGState, DocumentRAGAgent
from agents.document_parsing_agent import DocumentLayoutParsingState, DocumentParsingAgent
from pathlib import Path
import os
from utils.logger import AsyncLogger
from utils.config import API_HOST, API_PORT, WEBSOCKET_URL, LOG_LEVEL, ENDPOINT_TYPE, WORKSPACE_A, WORKSPACE_B, PLANNING_THRESHOLD, REFLECTION_ENABLED
from utils.llm_client import LLMClient
from utils.agent_team import AgentTeam
from utils.hierarchical_planner import HierarchicalPlanner
from utils.reflective_agent import ReflectiveAgent
from utils.agent_communication import AgentCommunicator
from utils.tools import ToolKit
from utils.memory_store import MemoryStore
from pydantic import BaseModel
from contextlib import asynccontextmanager
from agent_registry import AgentRegistry
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.communicator = AgentCommunicator()
    app.state.toolkit = ToolKit()
    app.state.memory_store = MemoryStore()
    yield

app = FastAPI(title="Agentic RAG Framework API", version="0.1.0", lifespan=lifespan)

origins = [
    "http://localhost:4200",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    document_path: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    database_connection: Optional[str] = None
    feedback: Optional[str] = None

class KnowledgeBaseEntry(BaseModel):
    id: str
    paths: List[str] = []
    databases: List[str] = []

knowledge_bases: dict[str, dict] = {}

@app.post("/knowledge_base")
async def create_knowledge_base(entries: List[KnowledgeBaseEntry]):
    for entry in entries:
        knowledge_bases[entry.id] = {"paths": entry.paths, "databases": entry.databases}
        for path in entry.paths:
            if Path(path).is_file() and path.endswith('.pdf'):
                parser = AgentRegistry.get_agent("parsing_agent")()
                state = DocumentLayoutParsingState(document_path=str(Path(path)))
                parsed = await parser.graph.ainvoke(state)
                rag_agent = AgentRegistry.get_agent("rag_agent")()
                rag_state = DocumentRAGState(
                    question="Initialize knowledge base",
                    document_path=str(Path(path)),
                    pages_as_base64_jpeg_images=parsed["pages_as_base64_jpeg_images"],
                    pages_as_text=parsed["pages_as_text"],
                    documents=parsed["documents"]
                )
                await rag_agent.graph.ainvoke(rag_state)
    await AsyncLogger.info(f"Knowledge base created/updated for IDs: {[entry.id for entry in entries]}")
    return {"message": "Knowledge base created/updated", "id": [entry.id for entry in entries]}

@app.post("/upload_and_query")
async def upload_and_query(file: UploadFile = File(...), question: Optional[str] = None, summarize: bool = False, team_mode: bool = False, parallel_execution: bool = False, require_human_review: bool = False, background_tasks: BackgroundTasks = None):
    if not question and not summarize:
        raise HTTPException(status_code=400, detail="Question or summarize flag is required")

    start_time = time.time()
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            await f.write(await file.read())

        llm_client = LLMClient(endpoint_type=ENDPOINT_TYPE, workspace_a=WORKSPACE_A, workspace_b=WORKSPACE_B)
        communicator = app.state.communicator
        toolkit = app.state.toolkit
        memory_store = app.state.memory_store
        planner = HierarchicalPlanner(llm_client, communicator, toolkit, memory_store)
        reflective_agent = ReflectiveAgent(llm_client, communicator, memory_store)

        multimodal_context = {"text": [f"File: {file.filename}"], "images": []} if file_path else None
        if team_mode:
            team = AgentTeam(["parsing_agent", "rag_agent"], llm_client, communicator, toolkit)
            if len(question or "") > PLANNING_THRESHOLD:
                plan = await planner.plan_task(f"Process PDF: {question or 'Summarize'}", multimodal_context)
                optimized_plan = await planner.optimize_plan(plan, multimodal_context)
                result = await planner.execute_plan(optimized_plan, team, parallel_tasks=parallel_execution, require_human_review=require_human_review)
                synthesized = await llm_client.generate(f"Synthesize results: {result}")
                final_result = {"response": synthesized}
                if REFLECTION_ENABLED and background_tasks:
                    background_tasks.add_task(reflective_agent.reflect, question, synthesized, "User feedback", multimodal_context)
            else:
                state = {"document_path": file_path, "query": question, "summarize": summarize}
                result = await team.execute_task(state, parallel=parallel_execution)
                final_result = result
        else:
            router = AgentRegistry.get_agent("router_agent")()
            state = RouterState(
                document_path=file_path,
                query=question,
                summarize=summarize,
                multimodal_context=multimodal_context
            )
            result = await router.graph.ainvoke(state)
            final_result = result["result"]

        if REFLECTION_ENABLED and background_tasks:
            reflection = await reflective_agent.reflect(question, final_result["response"], "Initial feedback", multimodal_context)
            refined_strategy = await reflective_agent.apply_improvements(reflection, "Basic response", router)
            background_tasks.add_task(AsyncLogger.info, f"Refined strategy: {refined_strategy}")

        await AsyncLogger.info(f"Processed query for {file.filename}: {final_result}")
        duration = time.time() - start_time
        await AsyncLogger.info(f"Endpoint /upload_and_query processed in {duration:.2f} seconds")
        return final_result
    except Exception as e:
        await AsyncLogger.error(f"Error processing upload_and_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_time = time.time()
    try:
        while True:
            data = await websocket.receive_json()
            msg = ChatMessage(**data)
            llm_client = LLMClient(endpoint_type=ENDPOINT_TYPE, workspace_a=WORKSPACE_A, workspace_b=WORKSPACE_B)
            communicator = app.state.communicator
            toolkit = app.state.toolkit
            memory_store = app.state.memory_store
            router = AgentRegistry.get_agent("router_agent")()
            reflective_agent = ReflectiveAgent(llm_client, communicator, memory_store)
            planner = HierarchicalPlanner(llm_client, communicator, toolkit, memory_store)
            multimodal_context = {"text": [msg.message], "images": []} if msg.document_path else None

            state = RouterState(
                document_path=msg.document_path,
                query=msg.message if "what" in msg.message.lower() or "?" in msg.message else None,
                summarize="summarize" in msg.message.lower(),
                knowledge_base_id=msg.knowledge_base_id,
                database_connection=msg.database_connection,
                feedback=msg.feedback,
                multimodal_context=multimodal_context
            )
            await AsyncLogger.info(f"Received chat message: {msg.message}")

            async def stream_response():
                team = AgentTeam(["parsing_agent", "rag_agent"], llm_client, communicator, toolkit) if msg.document_path else None
                if team and len(msg.message or "") > PLANNING_THRESHOLD:
                    plan = await planner.plan_task(msg.message, multimodal_context)
                    async for token in planner.stream_plan(msg.message, multimodal_context):
                        await websocket.send_text(token)
                    optimized_plan = await planner.optimize_plan(plan, multimodal_context)
                    async for token in planner.execute_plan_stream(optimized_plan, team, parallel_tasks=True, require_human_review=True):
                        await websocket.send_text(token)
                    result = await planner.execute_plan(optimized_plan, team, parallel_tasks=True, require_human_review=True)
                    reflection = await reflective_agent.reflect(msg.message, result[0], msg.feedback, multimodal_context)
                    refined_strategy = await reflective_agent.apply_improvements(reflection, "Chat strategy", router)
                    await websocket.send_text(f"Reflection applied: {refined_strategy}\n")
                else:
                    async for token in router.graph.ainvoke(state, start_node="stream_response"):
                        await websocket.send_text(token)
                await websocket.send_json({"end": True})

            # Handle human review responses
            async def handle_human_review():
                while True:
                    messages = await communicator.receive_messages("hierarchical_planner")
                    for msg in messages:
                        if msg.sender == "hierarchical_planner" and "Review required" in msg.content:
                            await websocket.send_text(f"{msg.content} (Reply with 'yes' or 'no')")
                            response = await websocket.receive_text()
                            await communicator.send_message("human", "hierarchical_planner", response, {"response": response})
                    await asyncio.sleep(1)

            # Run response streaming and human review handling concurrently
            await asyncio.gather(stream_response(), handle_human_review())
            duration = time.time() - start_time
            await AsyncLogger.info(f"Endpoint /chat processed in {duration:.2f} seconds")
    except Exception as e:
        await AsyncLogger.error(f"Error in chat endpoint: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        duration = time.time() - start_time
        await AsyncLogger.info(f"Endpoint /chat completed in {duration:.2f} seconds")

@app.post("/feedback")
async def submit_feedback(query: str, response: str, rating: int, comment: Optional[str] = None):
    start_time = time.time()
    await AsyncLogger.info(f"Feedback - Query: {query}, Response: {response}, Rating: {rating}, Comment: {comment}")
    llm_client = LLMClient(endpoint_type=ENDPOINT_TYPE, workspace_a=WORKSPACE_A, workspace_b=WORKSPACE_B)
    reflective_agent = ReflectiveAgent(llm_client, app.state.communicator, app.state.memory_store)
    multimodal_context = {"text": [query], "images": []}
    reflection = await reflective_agent.reflect(query, response, f"Rating: {rating}, Comment: {comment}", multimodal_context)
    router = AgentRegistry.get_agent("router_agent")()
    state = RouterState(query=query, result={"response": response}, feedback=f"Rating: {rating}, Comment: {comment}", multimodal_context=multimodal_context)
    result = await router.graph.ainvoke(state)
    refined_strategy = await reflective_agent.apply_improvements(reflection, "Feedback strategy", router)
    duration = time.time() - start_time
    await AsyncLogger.info(f"Endpoint /feedback processed in {duration:.2f} seconds")
    return {"message": "Feedback received and processed", "reflection": result["result"].get("reflection", {}), "improvement": refined_strategy}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, workers=2)
