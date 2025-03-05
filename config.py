"""Configuration settings for the agentic RAG framework."""

import os
from dotenv import load_dotenv
import torch

# Load environment variables from .env file if present
load_dotenv()

# LLM Configuration
LLM_MODE = os.getenv("LLM_MODE", "api").lower()  # Options: "api" or "local" (default to API mode)

# API Configuration (used if LLM_MODE == "api")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://1rche002papd.sdi.corp.bankofamerica.com:8005/v1/chat/completions")
MODEL_NAME_API = os.getenv("MODEL_NAME_API", "Meta-Llama-3-8B-Instruct")
TOKEN = os.getenv("PHOENIX_API_TOKEN", "abcdefghi")

# Local Configuration (used if LLM_MODE == "local")
MODEL_NAME_LOCAL = os.getenv("MODEL_NAME_LOCAL", "meta-llama/Llama-3-8B-Instruct")  # Adjust based on GPU capability
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, fallback to CPU
MULTIMODAL_MODEL_LOCAL = os.getenv("MULTIMODAL_MODEL_LOCAL", "Salesforce/blip-image-captioning-base")  # For image processing

# Common LLM Parameters
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1000"))  # Maximum tokens for LLM responses
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))  # Creativity of responses
TOP_P = float(os.getenv("TOP_P", "0.9"))  # Nucleus sampling
STOP_SEQUENCES = os.getenv("STOP_SEQUENCES", "\n").split(",")  # Stop sequences for generation

# Embedding Configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # Default embedding model for RAG
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))  # Number of documents to retrieve in RAG

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # Logging verbosity (DEBUG, INFO, WARNING, ERROR)
LOG_FORMAT = os.getenv("LOG_FORMAT", "{time} {level} {message}")  # Loguru format string
LOG_FILE = os.getenv("LOG_FILE", "agentic_rag.log")  # Optional log file path

# Database Configuration
DATABASE_DEFAULT = os.getenv("DATABASE_DEFAULT", "sqlite:///path/to/database.db")  # Default database connection string

# WebSocket and API Configuration
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/chat")  # WebSocket endpoint for chat
API_HOST = os.getenv("API_HOST", "0.0.0.0")  # FastAPI host
API_PORT = int(os.getenv("API_PORT", "8000"))  # FastAPI port

# Multimodal Configuration
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "5"))  # Maximum number of images to process in multimodal RAG
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "85"))  # JPEG quality for base64 encoding (0-100)

# Agentic Reasoning Configuration
PLANNING_THRESHOLD = int(os.getenv("PLANNING_THRESHOLD", "20"))  # Minimum query length to trigger hierarchical planning
REFLECTION_ENABLED = os.getenv("REFLECTION_ENABLED", "true").lower() == "true"  # Enable self-reflection

# Validate configurations
if LLM_MODE not in ["api", "local"]:
    raise ValueError("LLM_MODE must be 'api' or 'local'")

if not (0 <= TEMPERATURE <= 1.0):
    raise ValueError("TEMPERATURE must be between 0 and 1")

if not (0 <= TOP_P <= 1.0):
    raise ValueError("TOP_P must be between 0 and 1")

if not (0 <= IMAGE_QUALITY <= 100):
    raise ValueError("IMAGE_QUALITY must be between 0 and 100")

# Configure loguru logger (ensuring compatibility with utils.logger)
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format=LOG_FORMAT)
if LOG_FILE:
    logger.add(LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT, rotation="500 MB", retention="10 days")
