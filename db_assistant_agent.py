from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator
from utils.llm_client import LLMClient
from utils.config import LOG_LEVEL, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES
from document_utils import extract_text_from_pdf
from hybrid_retriever import MultimodalRetriever
from langchain_core.documents import Document
import sqlite3
import asyncio
import json
from agent_registry import agent_registry
from document_ai_agents.logger import logger

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class DBAssistantState(BaseModel):
    question: str
    database_connection: Optional[str] = None
    document_paths: List[str] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    relevant_documents: List[Document] = Field(default_factory=list)
    sql_query: Optional[str] = None
    db_data: Optional[List[dict]] = None
    response: Optional[str] = None
    multimodal_context: Optional[dict] = Field(None, description="Text and image data from documents")  # For multimodal support

@agent_registry.register(
    name="db_assistant_agent",
    capabilities=["qa"],
    supported_doc_types=["database", "generic"]
)
class DBAssistantAgent:
    def __init__(self, embed_model="all-MiniLM-L6-v2", k=3):
        self.llm_client = LLMClient()
        self.embedder = SentenceTransformer(embed_model)
        self.retriever = None
        self.k = k
        self.graph = self.build_agent()

    async def preprocess_documents(self, state: DBAssistantState):
        all_documents = []
        tasks = []
        for path in state.document_paths:
            if Path(path).is_file() and path.endswith('.pdf'):
                task = asyncio.create_task(asyncio.to_thread(extract_text_from_pdf, path))
                tasks.append(task)
        text_pages_list = await asyncio.gather(*tasks)
        for path, text_pages in zip(state.document_paths, text_pages_list):
            if Path(path).is_file() and path.endswith('.pdf'):
                all_documents.extend([
                    Document(page_content=page, metadata={"page_number": i, "document_path": path})
                    for i, page in enumerate(text_pages)
                ])
        
        # Extract multimodal context (text + images) using DocumentParsingAgent
        parser_tasks = []
        for path in state.document_paths:
            if Path(path).is_file() and path.endswith('.pdf'):
                parse_state = DocumentLayoutParsingState(document_path=path)
                parser = agent_registry.get_agent("parsing_agent")()
                parser_tasks.append(parser.graph.ainvoke(parse_state))
        parsed_results = await asyncio.gather(*parser_tasks)
        state.multimodal_context = {
            "text": [item for result in parsed_results for item in result["pages_as_text"]],
            "images": [item for result in parsed_results for item in result["pages_as_base64_jpeg_images"]]
        }
        state.documents = all_documents
        return state

    async def index_documents(self, state: DBAssistantState):
        self.retriever = MultimodalRetriever(state.documents, embedder=self.embedder, k=self.k)
        return state

    async def generate_sql_query(self, state: DBAssistantState):
        # Use multimodal context for better SQL generation if available
        text_context = "\n".join(state.multimodal_context.get("text", [])[:500])
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in state.multimodal_context.get("images", [])[:5]])
        prompt = (
            f"Given the user question: '{state.question}', and the following context:\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Generate a SQL query to fetch relevant data from a database. Assume a simple relational database with common tables like 'users', 'transactions', 'products', etc. "
            "Return only the SQL query as a string, no explanation."
        )
        sql_query = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not state.multimodal_context["images"] else state.multimodal_context["images"][0], max_length=200, temperature=0.1, top_p=0.9, stop=STOP_SEQUENCES)
        state.sql_query = sql_query.strip()
        return state

    async def query_database(self, state: DBAssistantState):
        if not state.database_connection or not state.sql_query:
            return state

        try:
            async with asyncio.to_thread(sqlite3.connect, state.database_connection) as conn:
                async with asyncio.to_thread(conn.cursor) as cursor:
                    await asyncio.to_thread(cursor.execute, state.sql_query)
                    columns = [description[0] for description in cursor.description]
                    rows = await asyncio.to_thread(cursor.fetchall)
                    state.db_data = [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            state.db_data = None
        return state

    async def combine_and_answer(self, state: DBAssistantState):
        context = ""
        if state.db_data:
            context += "Database Data:\n" + "\n".join([str(row) for row in state.db_data]) + "\n"
        if state.documents:
            text_context = "\n".join([doc.page_content for doc in state.documents if "Text-block" in doc.metadata.get("element_type", "")])
            image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, doc.metadata.get("image_base64", "")) for doc in state.documents if "Image" in doc.metadata.get("element_type", "")])
            context += f"Document Text: {text_context}\nDocument Images: {image_context}\n"

        # Use agentic reasoning for structured response
        prompt = (
            f"Question: {state.question}\n"
            f"Context: {context}\n"
            "Provide a detailed answer, integrating database data, document text, and image information:"
        )
        response = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not state.multimodal_context["images"] else state.multimodal_context["images"][0])
        state.response = response
        return state

    async def stream_response(self, state: DBAssistantState) -> AsyncGenerator[str, None]:
        """Stream the QA response token by token for real-time chat."""
        text_context = "\n".join(state.multimodal_context.get("text", [])[:500])
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in state.multimodal_context.get("images", [])[:5]])
        context = ""
        if state.db_data:
            context += "Database Data:\n" + "\n".join([str(row) for row in state.db_data]) + "\n"
        context += f"Document Text: {text_context}\nDocument Images: {image_context}\n"

        prompt = f"Question: {state.question}\nContext: {context}\nAnswer:"
        async for token in self.llm_client.generate_stream(prompt, image_base64=None if not state.multimodal_context["images"] else state.multimodal_context["images"][0]):
            yield token

    def build_agent(self):
        builder = StateGraph(DBAssistantState)
        builder.add_node("preprocess_documents", self.preprocess_documents)
        builder.add_node("index_documents", self.index_documents)
        builder.add_node("generate_sql_query", self.generate_sql_query)
        builder.add_node("query_database", self.query_database)
        builder.add_node("combine_and_answer", self.combine_and_answer)
        builder.add_node("stream_response", self.stream_response)
        
        builder.add_edge(START, "preprocess_documents")
        builder.add_edge("preprocess_documents", "index_documents")
        builder.add_edge("index_documents", "generate_sql_query")
        builder.add_edge("generate_sql_query", "query_database")
        builder.add_edge("query_database", "combine_and_answer")
        builder.add_conditional_edges(
            "combine_and_answer",
            lambda x: "stream_response" if x.response else END,
            path_map={"stream_response": END}
        )
        return builder.compile()
