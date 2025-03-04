from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List, Optional, AsyncGenerator
from langchain_core.documents import Document
import asyncio
from utils.logger import PerformanceLogger, AsyncLogger
from utils.config import EMBED_MODEL, RETRIEVAL_K, DEVICE, LOG_LEVEL
from document_ai_agents.logger import logger
import time

# Set log level from config
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class MultimodalRetriever:
    """
    A hybrid retriever for multimodal RAG, combining dense (embedding-based) and sparse (BM25) retrieval
    for text, with dense retrieval for images using CLIP.
    """
    def __init__(self, documents: List[Document], embedder: Optional[SentenceTransformer] = None, k: int = RETRIEVAL_K):
        """
        Initialize the MultimodalRetriever with documents and embedding models.
        
        Args:
            documents (List[Document]): List of LangChain Document objects (text and image metadata).
            embedder (Optional[SentenceTransformer]): SentenceTransformer for text embeddings (defaults to config EMBED_MODEL).
            k (int): Number of documents to retrieve (defaults to config RETRIEVAL_K).
        """
        self.documents = documents
        self.k = k
        self.perf_logger = PerformanceLogger()
        
        # Initialize text embedding model
        self.text_embedder = embedder or SentenceTransformer(EMBED_MODEL)
        self.text_docs = [doc for doc in documents if "Text-block" in doc.metadata.get("element_type", "")]
        self.text_bm25 = BM25Okapi([doc.page_content.split() for doc in self.text_docs])
        self.text_embeddings = self.text_embedder.encode([doc.page_content for doc in self.text_docs])
        
        # Initialize image embedding model (CLIP)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.image_docs = [doc for doc in documents if "Image" in doc.metadata.get("element_type", "")]
        self.image_embeddings = self.embed_images([doc.metadata.get("image_base64", "") for doc in self.image_docs])

    def embed_images(self, image_base64_list: List[Optional[str]]) -> np.ndarray:
        """
        Embed images using CLIP for dense retrieval.
        
        Args:
            image_base64_list (List[Optional[str]]): List of base64-encoded image strings or None.
        
        Returns:
            np.ndarray: Array of image embeddings.
        """
        embeddings = []
        for base64 in image_base64_list:
            if base64:
                try:
                    image = base64_to_pil_image(base64)
                    inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
                    outputs = self.clip_model.get_image_features(**inputs)
                    embeddings.append(outputs.detach().cpu().numpy()[0])
                except Exception as e:
                    logger.warning(f"Failed to embed image: {e}")
        return np.array(embeddings) if embeddings else np.array([])

    async def retrieve(self, query: str) -> List[Document]:
        """
        Asynchronously retrieve relevant documents using hybrid text retrieval and dense image retrieval.
        
        Args:
            query (str): The query string for retrieval.
        
        Returns:
            List[Document]: List of relevant documents (text and images), sorted by relevance.
        """
        self.perf_logger.start()
        start_time = time.time()
        try:
            # Text retrieval (dense + sparse)
            query_embedding = await asyncio.to_thread(self.text_embedder.encode, query)
            dense_text_scores = np.dot(self.text_embeddings, query_embedding)
            dense_text_indices = np.argsort(dense_text_scores)[::-1][:self.k // 2]
            dense_text_docs = [self.text_docs[i] for i in dense_text_indices]

            tokenized_query = query.split()
            bm25_scores = self.text_bm25.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[::-1][:self.k // 2]
            bm25_docs = [self.text_docs[i] for i in bm25_indices]

            text_results = list(dict.fromkeys(dense_text_docs + bm25_docs))[:self.k // 2]

            # Image retrieval (dense only, using CLIP)
            query_image_embedding = self.clip_model.get_text_features(**self.clip_processor(text=[query], return_tensors="pt").to(DEVICE)).detach().cpu().numpy()[0]
            if self.image_embeddings.size > 0:
                dense_image_scores = np.dot(self.image_embeddings, query_image_embedding)
                dense_image_indices = np.argsort(dense_image_scores)[::-1][:self.k // 2]
                image_results = [self.image_docs[i] for i in dense_image_indices]
            else:
                image_results = []

            # Combine text and image results, prioritizing relevance
            combined = list(dict.fromkeys(text_results + image_results))
            result = combined[:self.k]

            duration = time.time() - start_time
            await AsyncLogger.info(f"Retrieved {len(result)} documents in {duration:.2f} seconds for query: {query}")
            await self.perf_logger.async_stop("multimodal_retriever")
            return result
        except Exception as e:
            await AsyncLogger.error(f"Error in retrieval for query {query}: {e}")
            return []

    async def stream_retrieval(self, query: str) -> AsyncGenerator[str, None]:
        """
        Stream retrieval results token by token for real-time display.
        
        Args:
            query (str): The query string for retrieval.
        
        Yields:
            str: Tokens describing retrieved documents as they are processed.
        """
        documents = await self.retrieve(query)
        yield f"Retrieving documents for query: {query}\n"
        for i, doc in enumerate(documents, 1):
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            yield f"Document {i}/{len(documents)} - Type: {doc.metadata.get('element_type', 'Text-block')}, Content: {content}\n"
            if "Image" in doc.metadata.get("element_type", "") and doc.metadata.get("image_base64"):
                yield f"  Image description: {await asyncio.to_thread(self.llm_client.process_image, doc.metadata['image_base64'])}\n"
            await asyncio.sleep(0.1)  # Simulate streaming delay
        yield "Retrieval complete.\n"

    def validate_documents(self, documents: List[Document]) -> bool:
        """
        Validate the document list for consistency and required metadata.
        
        Args:
            documents (List[Document]): List of documents to validate.
        
        Returns:
            bool: True if documents are valid, False otherwise.
        """
        if not isinstance(documents, list):
            logger.warning("Documents must be a list")
            return False
        for doc in documents:
            if not isinstance(doc, Document) or not hasattr(doc, "page_content") or not hasattr(doc, "metadata"):
                logger.warning(f"Invalid document format: {doc}")
                return False
            if "element_type" not in doc.metadata:
                logger.warning(f"Missing element_type in metadata for document: {doc}")
                return False
        return True
