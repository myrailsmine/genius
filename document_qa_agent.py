from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator
from utils.llm_client import LLMClient
from utils.config import LOG_LEVEL, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES
from document_ai_agents.logger import logger
import asyncio
import json

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class AnswerChainOfThoughts(BaseModel):
    rationale: str
    relevant_context: str
    answer: str

class DocumentQAState(BaseModel):
    question: str
    pages_as_text: list[str] = Field(default_factory=list)
    pages_as_images: list[str] = Field(default_factory=list)  # Base64 JPEG images for multimodal support
    answer_cot: Optional[AnswerChainOfThoughts] = None

@agent_registry.register(
    name="qa_agent",
    capabilities=["qa"],
    supported_doc_types=["generic", "research_paper"]
)
class DocumentQAAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.graph = self.build_agent()

    async def answer_question(self, state: DocumentQAState) -> dict:
        # Combine text and image context for multimodal RAG
        text_context = "\n".join(state.pages_as_text[:5])  # Limit context for brevity
        image_context = "\n".join([await asyncio.to_thread(self.llm_client.process_image, img) for img in state.pages_as_images[:5]])
        
        # Use hierarchical reasoning to generate a structured response
        prompt = (
            f"Question: {state.question}\n"
            f"Text Context: {text_context}\n"
            f"Image Context: {image_context}\n"
            "Provide a detailed Chain of Thought (CoT) response with rationale, relevant_context, and answer in JSON format: "
            f"{AnswerChainOfThoughts.model_json_schema()}"
        )
        response = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=None if not state.pages_as_images else state.pages_as_images[0])
        
        try:
            import json
            answer_cot = AnswerChainOfThoughts(**json.loads(response))
        except Exception as e:
            logger.error(f"Failed to parse CoT response: {e}")
            answer_cot = AnswerChainOfThoughts(
                rationale="Error in processing",
                relevant_context=f"Text: {text_context}\nImages: {image_context}",
                answer="Unable to generate answer due to parsing error"
            )
        
        return {"answer_cot": answer_cot}

    async def stream_answer(self, state: DocumentQAState) -> AsyncGenerator[str, None]:
        """Stream the QA response token by token for real-time chat."""
        async for token in self.llm_client.generate_stream(
            f"Question: {state.question}\nText Context: {'\n'.join(state.pages_as_text[:5])}\nImage Context: {'\n'.join(await asyncio.gather(*[asyncio.to_thread(self.llm_client.process_image, img) for img in state.pages_as_images[:5]]))}\nAnswer:",
            image_base64=None if not state.pages_as_images else state.pages_as_images[0]
        ):
            yield token

    def build_agent(self):
        builder = StateGraph(DocumentQAState)
        builder.add_node("answer_question", self.answer_question)
        builder.add_node("stream_answer", self.stream_answer)
        builder.add_edge(START, "answer_question")
        builder.add_conditional_edges(
            "answer_question",
            lambda x: "stream_answer" if x.answer_cot else END,
            path_map={"stream_answer": END}
        )
        return builder.compile()
