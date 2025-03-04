from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Optional, Annotated, List, Callable
from utils.llm_client import LLMClient
from utils.config import LOG_LEVEL, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES
from document_utils import extract_text_from_pdf
from hybrid_retriever import MultimodalRetriever
from langchain_core.documents import Document
from agent_registry import agent_registry, tool_registry
from tools import search_wikipedia, search_duck_duck_go, get_page_content  # Example tools from tools.py
from document_ai_agents.logger import logger
import asyncio
import json

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class MultiToolState(BaseModel):
    document_path: Optional[str] = None
    question: Optional[str] = None
    summarize: bool = False
    documents: List[Document] = Field(default_factory=list)
    relevant_documents: List[Document] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    tool_results: dict = Field(default_factory=dict)
    response: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    database_connection: Optional[str] = None

@agent_registry.register(
    name="document_multi_tool_agent",
    capabilities=["qa", "summarization", "tool_use"],
    supported_doc_types=["generic", "term_sheet", "research_paper", "database"]
)
class DocumentMultiToolAgent:
    def __init__(self, embed_model="all-MiniLM-L6-v2", k=3):
        self.llm_client = LLMClient()
        self.embedder = SentenceTransformer(embed_model)
        self.retriever = None
        self.k = k
        self.available_tools = {
            name: tool for name, tool in tool_registry.tools.items()
            if name in ["search_wikipedia", "search_duck_duck_go", "get_page_content"]
        }
        self.graph = self.build_agent()

    def preprocess_document(self, state: MultiToolState):
        documents = []
        if state.document_path and Path(state.document_path).is_file() and state.document_path.endswith('.pdf'):
            text_pages = extract_text_from_pdf(state.document_path)
            documents.extend([
                Document(page_content=page, metadata={"page_number": i, "document_path": state.document_path})
                for i, page in enumerate(text_pages)
            ])
        elif state.knowledge_base_id:
            from api import knowledge_bases
            for path in knowledge_bases.get(state.knowledge_base_id, {}).get("paths", []):
                if Path(path).is_file() and path.endswith('.pdf'):
                    text_pages = extract_text_from_pdf(path)
                    documents.extend([
                        Document(page_content=page, metadata={"page_number": i, "document_path": path})
                        for i, page in enumerate(text_pages)
                    ])
        state.documents = documents
        return state

    def index_documents(self, state: MultiToolState):
        self.retriever = MultimodalRetriever(state.documents, embedder=self.embedder, k=self.k)
        return state

    async def select_tools(self, state: MultiToolState):
        if not state.question:
            return state
        prompt = (
            f"Given the query: '{state.question}', select the best tools from {list(self.available_tools.keys())} "
            "to assist in answering or summarizing. Return a JSON list of tool names."
        )
        tool_selection = await asyncio.to_thread(self.llm_client.generate, prompt, max_length=200, temperature=0.1, top_p=0.9, stop=STOP_SEQUENCES)
        try:
            tools = json.loads(tool_selection)
            state.tools_used = [t for t in tools if t in self.available_tools]
        except Exception as e:
            logger.error(f"Failed to parse tool selection: {e}")
            state.tools_used = []
        return state

    async def execute_tools(self, state: MultiToolState):
        if not state.tools_used or not state.question:
            return state
        tool_results = {}
        for tool_name in state.tools_used:
            tool = self.available_tools[tool_name]
            try:
                result = await asyncio.to_thread(tool, state.question)
                tool_results[tool_name] = result if isinstance(result, str) else str(result)
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                tool_results[tool_name] = f"Error: {str(e)}"
        state.tool_results = tool_results
        return state

    def combine_context(self, state: MultiToolState):
        relevant_docs = self.retriever.retrieve(state.question or "")
        text_context = "\n".join([doc.page_content for doc in relevant_docs if "Text-block" in doc.metadata.get("element_type", "")])
        image_context = "\n".join([self.llm_client.process_image(doc.metadata.get("image_base64", "")) for doc in relevant_docs if "Image" in doc.metadata.get("element_type", "")])
        tool_context = "\n".join([f"{tool}: {result}" for tool, result in state.tool_results.items()])
        context = f"Document Text: {text_context}\nDocument Images: {image_context}\nTool Results: {tool_context}"
        return context

    async def answer_question(self, state: MultiToolState):
        if not state.question and not state.summarize:
            return state
        context = self.combine_context(state)
        prompt = (
            f"{'Summarize' if state.summarize else 'Answer the question'}: {state.question}\n"
            f"Context: {context}\n"
            "Response:"
        )
        response = await asyncio.to_thread(self.llm_client.generate, prompt, max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP_SEQUENCES)
        state.response = response
        return state

    def query_database(self, state: MultiToolState):
        if not state.database_connection or not state.question:
            return state
        db_agent = agent_registry.get_agent("db_assistant_agent")()
        db_state = DBAssistantState(
            question=state.question,
            database_connection=state.database_connection,
            document_paths=[state.document_path] if state.document_path else []
        )
        db_result = db_agent.graph.invoke(db_state)
        if db_result.db_data:
            state.tool_results["database_query"] = str(db_result.db_data)
        return state

    def build_agent(self):
        builder = StateGraph(MultiToolState)
        builder.add_node("preprocess_document", self.preprocess_document)
        builder.add_node("index_documents", self.index_documents)
        builder.add_node("select_tools", self.select_tools)
        builder.add_node("execute_tools", self.execute_tools)
        builder.add_node("query_database", self.query_database)
        builder.add_node("answer_question", self.answer_question)

        builder.add_edge(START, "preprocess_document")
        builder.add_edge("preprocess_document", "index_documents")
        builder.add_edge("index_documents", "select_tools")
        builder.add_edge("select_tools", "execute_tools")
        builder.add_edge("execute_tools", "query_database")
        builder.add_edge("query_database", "answer_question")
        builder.add_edge("answer_question", END)

        return builder.compile()
