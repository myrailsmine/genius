from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Optional, Annotated, List, AsyncGenerator
from document_utils import extract_text_from_pdf
from agent_registry import agent_registry
from utils.llm_client import LLMClient
from utils.config import LOG_LEVEL
from utils.hierarchical_planner import HierarchicalPlanner
from utils.reflective_agent import ReflectiveAgent
import numpy as np
from sklearn.preprocessing import softmax
import asyncio
import json

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class RouterState(BaseModel):
    document_path: Optional[str] = None
    query: Optional[str] = None
    summarize: bool = False
    documents: List[Document] = Field(default_factory=list)
    database_connection: Optional[str] = None
    selected_agent: Optional[str] = None
    result: dict = Field(default_factory=dict)
    knowledge_base_id: Optional[str] = None
    plan: Optional[List[Dict]] = None
    feedback: Optional[str] = None
    multimodal_context: Optional[dict] = None  # For multimodal data (text + images)

@agent_registry.register(
    name="router_agent",
    capabilities=["routing", "planning", "reflection"],
    supported_doc_types=["generic", "term_sheet", "research_paper", "database"]
)
class RouterAgent:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.llm_client = LLMClient()
        self.embedder = SentenceTransformer(embed_model)
        self.planner = HierarchicalPlanner()
        self.reflector = ReflectiveAgent()
        self.doc_labels = ["term_sheet", "research_paper", "generic", "database"]
        self.intent_labels = ["qa", "summarization", "generic", "planning", "multimodal"]
        self.centroids = {
            "term_sheet": self.embedder.encode("valuation cap equity investors terms conditions"),
            "research_paper": self.embedder.encode("abstract methodology results conclusion"),
            "generic": self.embedder.encode("general document text information"),
            "database": self.embedder.encode("sql query database revenue transactions")
        }
        self.graph = self.build_agent()

    async def preprocess_document(self, state: RouterState):
        documents = []
        if state.document_path and Path(state.document_path).is_file() and state.document_path.endswith('.pdf'):
            text_pages = await asyncio.to_thread(extract_text_from_pdf, state.document_path)
            documents.extend([
                Document(page_content=page, metadata={"page_number": i, "document_path": state.document_path})
                for i, page in enumerate(text_pages)
            ])
        elif state.knowledge_base_id:
            from api import knowledge_bases
            tasks = []
            for path in knowledge_bases.get(state.knowledge_base_id, {}).get("paths", []):
                if Path(path).is_file() and path.endswith('.pdf'):
                    task = asyncio.create_task(asyncio.to_thread(extract_text_from_pdf, path))
                    tasks.append(task)
            text_pages_list = await asyncio.gather(*tasks)
            for path, text_pages in zip(knowledge_bases.get(state.knowledge_base_id, {}).get("paths", []), text_pages_list):
                if Path(path).is_file() and path.endswith('.pdf'):
                    documents.extend([
                        Document(page_content=page, metadata={"page_number": i, "document_path": path})
                        for i, page in enumerate(text_pages)
                    ])
        state.documents = documents
        # Extract multimodal context (text + potential images from parsing)
        parser = agent_registry.get_agent("parsing_agent")()
        if state.document_path or state.knowledge_base_id:
            parse_state = DocumentLayoutParsingState(document_path=state.document_path or list(knowledge_bases.get(state.knowledge_base_id, {}).get("paths", []))[0] if state.knowledge_base_id else None)
            parsed = await parser.graph.ainvoke(parse_state)
            state.multimodal_context = {
                "text": parsed["pages_as_text"],
                "images": parsed["pages_as_base64_jpeg_images"]
            }
        return state

    async def determine_agent(self, state: RouterState):
        if not state.query:
            return {"selected_agent": "rag_agent"}
        
        # Initial classification and planning
        content = "\n".join(state.multimodal_context.get("text", [])[:1000]) if state.multimodal_context else ""
        query = state.query or "summarize this" if state.summarize else "tell me about this"

        # Check for multimodal intent or database queries
        if state.database_connection or any(keyword in query.lower() for keyword in ["sql", "database", "revenue", "transactions", "query"]):
            doc_type = "database"
        elif any(keyword in query.lower() for keyword in ["image", "figure", "table", "multimodal"]):
            doc_type = "generic"  # Handle multimodal queries
        else:
            doc_scores = await asyncio.to_thread(self.llm_client.classify, content[:512], self.doc_labels)
            doc_probs = softmax([doc_scores.get(label, 0) for label in self.doc_labels])

            doc_embedding = self.embedder.encode(content) if content else self.embedder.encode("generic document")
            embed_scores = {
                doc_type: np.dot(doc_embedding, centroid) / (np.linalg.norm(doc_embedding) * np.linalg.norm(centroid))
                for doc_type, centroid in self.centroids.items()
            }
            embed_probs = softmax([score for score in embed_scores.values()])

            combined_scores = {}
            for i, doc_type in enumerate(self.doc_labels):
                combined_scores[doc_type] = 0.7 * doc_probs[i] + 0.3 * embed_probs[i]
            doc_type = max(combined_scores, key=combined_scores.get, default="generic")

        # Enhanced intent detection with multimodal and planning support
        intent_scores = await asyncio.to_thread(self.llm_client.classify, query, self.intent_labels)
        intent_probs = softmax([intent_scores.get(label, 0) for label in self.intent_labels])
        intent = max(self.intent_labels, key=lambda x: intent_scores.get(x, 0))

        # Hierarchical planning for complex or multimodal queries
        if intent in ["planning", "multimodal"] or len(query.split()) > 20:  # Complex or multimodal queries trigger planning
            plan = await asyncio.to_thread(self.planner.plan_task, query)
            optimized_plan = await asyncio.to_thread(self.planner.optimize_plan, plan)
            state.plan = optimized_plan
            return {"selected_agent": "multi_agent_orchestrator"}  # Assume this exists or create it

        for agent_name, metadata in agent_registry.list_agents().items():
            if agent_name == "router_agent":
                continue
            if doc_type in metadata.supported_doc_types and intent in metadata.capabilities:
                return {"selected_agent": agent_name}
        return {"selected_agent": "rag_agent"}

    async def reflect_and_improve(self, state: RouterState):
        if state.feedback and state.result:
            reflection = await asyncio.to_thread(self.reflector.reflect, state.query, state.result.get("response", ""), state.feedback)
            # Apply improvements to agent strategies or store for learning
            logger.info(f"Reflection for query {state.query}: {reflection}")
            return {"result": {**state.result, "reflection": reflection}}
        return state

    async def route_to_agent(self, state: RouterState):
        agent_class = agent_registry.get_agent(state.selected_agent)
        if not agent_class:
            return {"result": {"error": f"Agent {state.selected_agent} not found"}}

        agent = agent_class()
        if state.selected_agent == "db_assistant_agent":
            agent_state = DBAssistantState(
                question=state.query or "Provide a summary",
                database_connection=state.database_connection,
                document_paths=[state.document_path] if state.document_path else [],
                documents=state.documents
            )
            result = await agent.graph.ainvoke(agent_state)
            return {"result": {"response": result.response}}
        elif state.selected_agent == "term_sheet_agent":
            agent_state = TermSheetState(
                document_path=state.document_path,
                question=state.query,
                summarize=state.summarize,
                documents=state.documents
            )
            result = await agent.graph.ainvoke(agent_state)
            return {"result": {"answer": result["answer"], "summary": result["summary"]}}
        elif state.selected_agent == "qa_agent":
            agent_state = DocumentQAState(
                question=state.query,
                pages_as_text=[doc.page_content for doc in state.documents if "Text-block" in doc.metadata.get("element_type", "")],
                pages_as_images=[doc.metadata.get("image_base64", "") for doc in state.documents if "Image" in doc.metadata.get("element_type", "")]
            )
            if state.query:  # Stream for chat
                async def stream_result():
                    async for token in agent.graph.ainvoke(agent_state, start_node="stream_answer"):
                        yield token
                return {"result": {"stream": stream_result()}}
            else:  # Return full response
                result = await agent.graph.ainvoke(agent_state)
                return {"result": {"answer": result["answer_cot"].answer}}
        elif state.selected_agent == "rag_agent":
            agent_state = DocumentRAGState(
                question=state.query or "Provide a summary",
                document_path=state.document_path,
                pages_as_base64_jpeg_images=[doc.metadata.get("image_base64", "") for doc in state.documents if "Image" in doc.metadata.get("element_type", "")],
                pages_as_text=[doc.page_content for doc in state.documents if "Text-block" in doc.metadata.get("element_type", "")],
                documents=state.documents,
                knowledge_base_id=state.knowledge_base_id
            )
            if state.query:  # Stream for chat
                async def stream_result():
                    async for token in agent.graph.ainvoke(agent_state, start_node="stream_answer"):
                        yield token
                return {"result": {"stream": stream_result()}}
            else:  # Return full response
                result = await agent.graph.ainvoke(agent_state)
                return {"result": {"response": result["response"]}}
        elif state.selected_agent == "document_multi_tool_agent":
            agent_state = MultiToolState(
                document_path=state.document_path,
                question=state.query,
                summarize=state.summarize,
                documents=state.documents,
                knowledge_base_id=state.knowledge_base_id,
                database_connection=state.database_connection
            )
            result = await agent.graph.ainvoke(agent_state)
            return {"result": {"response": result.response}}
        return {"result": {"error": "Unsupported agent"}}

    async def stream_response(self, state: RouterState) -> AsyncGenerator[str, None]:
        """Stream the routed agent's response for real-time chat."""
        result = await self.route_to_agent(state)
        if "stream" in result["result"]:
            async for token in result["result"]["stream"]:
                yield token
        else:
            yield result["result"].get("response", "") + "\n"

    def build_agent(self):
        builder = StateGraph(RouterState)
        builder.add_node("preprocess_document", self.preprocess_document)
        builder.add_node("determine_agent", self.determine_agent)
        builder.add_node("route_to_agent", self.route_to_agent)
        builder.add_node("reflect_and_improve", self.reflect_and_improve)
        builder.add_node("stream_response", self.stream_response)
        builder.add_edge(START, "preprocess_document")
        builder.add_edge("preprocess_document", "determine_agent")
        builder.add_edge("determine_agent", "route_to_agent")
        builder.add_conditional_edges(
            "route_to_agent",
            lambda x: "reflect_and_improve" if x.feedback and "response" in x.result else "stream_response",
            path_map={"reflect_and_improve": "stream_response", "stream_response": END}
        )
        builder.add_edge("reflect_and_improve", "stream_response")
        builder.add_edge("stream_response", END)
        return builder.compile()
