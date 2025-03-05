from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator, Optional
from agents.router_agent import RouterState, RouterAgent
from agents.document_rag_agent import DocumentRAGState, DocumentRAGAgent
from agents.document_parsing_agent import DocumentLayoutParsingState, DocumentParsingAgent
from pathlib import Path
import os
from utils.logger import AsyncLogger
from utils.config import API_HOST, API_PORT, WEBSOCKET_URL, LOG_LEVEL
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel

app = FastAPI(title="Agentic RAG Framework API", version="0.1.0")

# Add CORS middleware
origins = [
    "http://localhost:4200",  # Angular dev server
    "http://localhost:8000",  # FastAPI server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Initialize rate limiting
    await FastAPILimiter.init(app)

class ChatMessage(BaseModel):
    message: str
    document_path: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    database_connection: Optional[str] = None
    feedback: Optional[str] = None

class KnowledgeBaseEntry(BaseModel):
    id: str
    paths: List[str] = []  # PDF paths
    databases: List[str] = []  # Database connection strings

# In-memory storage for knowledge base (replace with a database in production)
knowledge_bases: dict[str, dict] = {}

@app.post("/knowledge_base")
async def create_knowledge_base(entries: List[KnowledgeBaseEntry]):
    """Create or update a knowledge base with PDF paths and databases."""
    for entry in entries:
        knowledge_bases[entry.id] = {"paths": entry.paths, "databases": entry.databases}
        for path in entry.paths:
            if Path(path).is_file() and path.endswith('.pdf'):
                parser = agent_registry.get_agent("parsing_agent")()
                state = DocumentLayoutParsingState(document_path=str(Path(path)))
                parsed = await parser.graph.ainvoke(state)
                rag_agent = agent_registry.get_agent("rag_agent")()
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

@app.post("/upload_and_query", dependencies=[Depends(RateLimiter(times=5, minutes=1))])
async def upload_and_query(file: UploadFile = File(...), question: Optional[str] = None, summarize: bool = False, database_connection: Optional[str] = None):
    """Upload a PDF and process a QA or summarization query with optional database integration."""
    if not question and not summarize:
        raise HTTPException(status_code=400, detail="Question or summarize flag is required")
    
    start_time = time.time()
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            await f.write(await file.read())
        
        router = agent_registry.get_agent("router_agent")()
        state = RouterState(
            document_path=file_path,
            query=question,
            summarize=summarize,
            database_connection=database_connection
        )
        result = await router.graph.ainvoke(state)
        await AsyncLogger.info(f"Processed query for {file.filename}: {result['result']}")
        duration = time.time() - start_time
        await AsyncLogger.info(f"Endpoint /upload_and_query processed in {duration:.2f} seconds")
        return result["result"]
    except Exception as e:
        await AsyncLogger.error(f"Error processing upload_and_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """Handle real-time chat with streaming responses, supporting multimodal and agentic queries."""
    await websocket.accept()
    start_time = time.time()
    try:
        while True:
            data = await websocket.receive_json()
            msg = ChatMessage(**data)
            router = agent_registry.get_agent("router_agent")()
            state = RouterState(
                document_path=msg.document_path,
                query=msg.message if "what" in msg.message.lower() or "?" in msg.message else None,
                summarize="summarize" in msg.message.lower(),
                knowledge_base_id=msg.knowledge_base_id,
                database_connection=msg.database_connection,
                feedback=msg.feedback
            )
            await AsyncLogger.info(f"Received chat message: {msg.message}")
            
            # Stream the response
            async def stream_response():
                async for token in router.graph.ainvoke(state, start_node="stream_response"):
                    await websocket.send_text(token)
                await websocket.send_json({"end": True})
            
            await stream_response()
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
    """Submit user feedback for agent improvement."""
    start_time = time.time()
    await AsyncLogger.info(f"Feedback - Query: {query}, Response: {response}, Rating: {rating}, Comment: {comment}")
    router = agent_registry.get_agent("router_agent")()
    state = RouterState(query=query, result={"response": response}, feedback=f"Rating: {rating}, Comment: {comment}")
    result = await router.graph.ainvoke(state)
    duration = time.time() - start_time
    await AsyncLogger.info(f"Endpoint /feedback processed in {duration:.2f} seconds")
    return {"message": "Feedback received and processed", "reflection": result["result"].get("reflection", {})}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
