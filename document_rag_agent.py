from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator
from hybrid_retriever import MultimodalRetriever
from utils.llm_client import LLMClient
from utils.config import LOG_LEVEL, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES
from document_ai_agents.logger import logger
import asyncio

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class DocumentRAGState(BaseModel):
    question: str
    document_path: Optional[str] = None
    pages_as_base64_jpeg_images: list[str] = Field(default_factory=list)
    pages_as_text: list[str] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    relevant_documents: list[Document] = Field(default_factory=list)
    response: Optional[str] = None
    knowledge_base_id: Optional[str] = None

@agent_registry.register(
    name="rag_agent",
    capabilities=["qa"],
    supported_doc_types=["generic"]
)
class DocumentRAGAgent:
    def __init__(self, embed_model="all-MiniLM-L6-v2", k=3):
        self.llm_client = LLMClient()
        self.embedder = SentenceTransformer(embed_model)
        self.retriever = None
        self.k = k
        self.graph = self.build_agent()

    async def index_documents(self, state: DocumentRAGState):
        if state.knowledge_base_id:
            from api import knowledge_bases
            all_documents = []
            tasks = []
            for path in knowledge_bases.get(state.knowledge_base_id, {}).get("paths", []):
                if Path(path).is_file() and path.endswith('.pdf'):
                    task = asyncio.create_task(self.process_single_document(path))
                    tasks.append(task)
            results = await asyncio.gather(*tasks)
            all_documents = [doc for result in results for doc in result]
            state.documents = all_documents
        elif state.document_path and Path(state.document_path).is_file() and state.document_path.endswith('.pdf'):
            parser = agent_registry.get_agent("parsing_agent")()
            parse_state = DocumentLayoutParsingState(document_path=state.document_path)
            parsed = await parser.graph.ainvoke(parse_state)
            state.documents = parsed["documents"]
            state.pages_as_base64_jpeg_images = parsed["pages_as_base64_jpeg_images"]
            state.pages_as_text = parsed["pages_as_text"]
        self.retriever = MultimodalRetriever(state.documents, embedder=self.embedder, k=self.k)

    async def process_single_document(self, path: str):
        parser = agent_registry.get_agent("parsing_agent")()
        parse_state = DocumentLayoutParsingState(document_path=path)
        parsed = await parser.graph.ainvoke(parse_state)
        return parsed["documents"]

    async def answer_question(self, state: DocumentRAGState) -> dict:
        relevant_docs = await asyncio.to_thread(self.retriever.retrieve, state.question)
        text_context = "\n".join([doc.page_content for doc in relevant_docs if "Text-block" in doc.metadata.get("element_type", "")])
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, doc.metadata.get("image_base64", "")) for doc in relevant_docs if "Image" in doc.metadata.get("element_type", "")])
        
        # Use agentic reasoning for structured response
        prompt = (
            f"Question: {state.question}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Provide a detailed answer, considering both text and image information:"
        )
        response = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not state.pages_as_base64_jpeg_images else state.pages_as_base64_jpeg_images[0])
        state.response = response
        return {"response": response, "relevant_documents": relevant_docs}

    async def stream_answer(self, state: DocumentRAGState) -> AsyncGenerator[str, None]:
        """Stream the RAG response token by token for real-time chat."""
        relevant_docs = await asyncio.to_thread(self.retriever.retrieve, state.question)
        text_context = "\n".join([doc.page_content for doc in relevant_docs if "Text-block" in doc.metadata.get("element_type", "")])
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, doc.metadata.get("image_base64", "")) for doc in relevant_docs if "Image" in doc.metadata.get("element_type", "")])
        
        prompt = (
            f"Question: {state.question}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Answer:"
        )
        async for token in self.llm_client.generate_stream(prompt, image_base64=None if not state.pages_as_base64_jpeg_images else state.pages_as_base64_jpeg_images[0]):
            yield token

    def build_agent(self):
        builder = StateGraph(DocumentRAGState)
        builder.add_node("index_documents", self.index_documents)
        builder.add_node("answer_question", self.answer_question)
        builder.add_node("stream_answer", self.stream_answer)
        builder.add_edge(START, "index_documents")
        builder.add_edge("index_documents", "answer_question")
        builder.add_conditional_edges(
            "answer_question",
            lambda x: "stream_answer" if x.response else END,
            path_map={"stream_answer": END}
        )
        return builder.compile()
