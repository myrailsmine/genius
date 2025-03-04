from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator
from document_utils import extract_text_from_pdf
from hybrid_retriever import MultimodalRetriever
from agent_registry import agent_registry
from utils.llm_client import LLMClient
from utils.config import LOG_LEVEL, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES
from document_ai_agents.logger import logger
import asyncio
import json

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class TermSheetState(BaseModel):
    document_path: str
    question: Optional[str] = None
    summarize: bool = False
    documents: List[Document] = Field(default_factory=list)
    relevant_documents: List[Document] = Field(default_factory=list)
    answer: Optional[str] = None
    summary: Optional[str] = None
    multimodal_context: Optional[dict] = Field(None, description="Text and image data from the document")  # For multimodal support

@agent_registry.register(
    name="term_sheet_agent",
    capabilities=["qa", "summarization"],
    supported_doc_types=["term_sheet"]
)
class TermSheetAgent:
    def __init__(self, embed_model="all-MiniLM-L6-v2", k=3):
        self.llm_client = LLMClient()
        self.embedder = SentenceTransformer(embed_model)
        self.retriever = None
        self.k = k
        self.graph = self.build_agent()

    async def preprocess_document(self, state: TermSheetState):
        assert Path(state.document_path).is_file() and state.document_path.endswith('.pdf'), "Invalid or non-PDF file"
        text_pages = await asyncio.to_thread(extract_text_from_pdf, state.document_path)
        documents = [
            Document(page_content=page, metadata={"page_number": i, "document_path": state.document_path})
            for i, page in enumerate(text_pages)
        ]
        state.documents = documents

        # Extract multimodal context (text + images) using DocumentParsingAgent
        parser = agent_registry.get_agent("parsing_agent")()
        parse_state = DocumentLayoutParsingState(document_path=state.document_path)
        parsed = await parser.graph.ainvoke(parse_state)
        state.multimodal_context = {
            "text": parsed["pages_as_text"],
            "images": parsed["pages_as_base64_jpeg_images"]
        }
        return state

    async def index_documents(self, state: TermSheetState):
        self.retriever = MultimodalRetriever(state.documents, embedder=self.embedder, k=self.k)
        return state

    async def answer_question(self, state: TermSheetState):
        if not state.question:
            return state
        relevant_docs = await asyncio.to_thread(self.retriever.retrieve, state.question)
        text_context = "\n".join([doc.page_content for doc in relevant_docs if "Text-block" in doc.metadata.get("element_type", "")])
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in [doc.metadata.get("image_base64", "") for doc in relevant_docs if "Image" in doc.metadata.get("element_type", "")]])
        
        # Use agentic reasoning for structured QA
        prompt = (
            f"Question: {state.question}\n"
            f"Text Context from Term Sheet: {text_context}\n"
            f"Image Context from Term Sheet: {image_context}\n"
            "Answer the question about this term sheet in a concise, precise manner, focusing on financial terms, valuation, or equity details:"
        )
        response = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not state.multimodal_context["images"] else state.multimodal_context["images"][0])
        state.answer = response
        return {"answer": response, "relevant_documents": relevant_docs}

    async def summarize_document(self, state: TermSheetState):
        if not state.summarize:
            return state
        full_text = "\n".join(state.multimodal_context["text"])
        image_summaries = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in state.multimodal_context["images"]])
        
        prompt = (
            f"Summarize the following term sheet:\nText: {full_text}\nImages: {image_summaries}\n"
            "Follow these guidelines:\n"
            "1. Extract and convey ONLY the precise summarization assistant stated in the source text and images.\n"
            "2. Maintain the original meaning and intent without adding external information.\n"
            "3. Present information in a clear, professional, and organized manner, focusing on financial terms, valuation, equity, and key conditions.\n"
            "Output format:\n"
            "A concise executive summary\n"
            "Key Points: [Main ideas explicitly stated in the text and images]\n"
            "Supporting Details: [Relevant evidence and specific information from the source]\n"
            "Note: If any part of the text or images is unclear or ambiguous, please maintain that ambiguity rather than making assumptions."
        )
        summary = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not state.multimodal_context["images"] else state.multimodal_context["images"][0], max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP_SEQUENCES)
        state.summary = summary
        return {"summary": summary}

    async def stream_response(self, state: TermSheetState) -> AsyncGenerator[str, None]:
        """Stream the QA or summary response token by token for real-time chat."""
        if state.question:
            relevant_docs = await asyncio.to_thread(self.retriever.retrieve, state.question)
            text_context = "\n".join([doc.page_content for doc in relevant_docs if "Text-block" in doc.metadata.get("element_type", "")])
            image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in [doc.metadata.get("image_base64", "") for doc in relevant_docs if "Image" in doc.metadata.get("element_type", "")]])
            prompt = f"Question: {state.question}\nText Context: {text_context}\nImage Context: {image_context}\nAnswer:"
        else:
            full_text = "\n".join(state.multimodal_context["text"])
            image_summaries = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in state.multimodal_context["images"]])
            prompt = (
                f"Summarize the following term sheet:\nText: {full_text}\nImages: {image_summaries}\n"
                "Provide a concise response:"
            )
        
        async for token in self.llm_client.generate_stream(prompt, image_base64=None if not state.multimodal_context["images"] else state.multimodal_context["images"][0]):
            yield token

    def build_agent(self):
        builder = StateGraph(TermSheetState)
        builder.add_node("preprocess_document", self.preprocess_document)
        builder.add_node("index_documents", self.index_documents)
        builder.add_node("answer_question", self.answer_question)
        builder.add_node("summarize_document", self.summarize_document)
        builder.add_node("stream_response", self.stream_response)
        
        builder.add_edge(START, "preprocess_document")
        builder.add_edge("preprocess_document", "index_documents")
        builder.add_conditional_edges(
            "index_documents",
            lambda x: "answer_question" if x.question else "summarize_document",
            path_map={"answer_question": "stream_response", "summarize_document": "stream_response"}
        )
        builder.add_edge("stream_response", END)
        
        return builder.compile()
