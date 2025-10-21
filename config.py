import os
from dotenv import load_dotenv
from typing import Dict, Any
from logger import logger

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for RAG 2.0 system, loaded from environment variables."""
    
    logger.info("Loading configuration...")
    
    # --- Model configurations ---
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # --- NEW LLM Configs ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "featherless-ai")
    LLM_MODEL = os.getenv("LLM_MODEL", "inclusionAI/Ling-1T")
    HF_API_TOKEN = os.getenv("HF_API_TOKEN") # Used as the api_key

    # --- Vector store settings ---
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma") # 'chroma' or 'faiss'
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_store")
    FAISS_DB_PATH = os.getenv("FAISS_DB_PATH", "./faiss_vector_store")
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    
    # --- Web crawler settings ---
    REQUEST_TIMEOUT = 30
    
    # --- UI settings ---
    PAGE_TITLE = "RAG 2.0 - Advanced Knowledge System"
    PAGE_ICON = "🧠"
    
    # File upload settings
    MAX_FILE_SIZE = 200  # MB
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md']

    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_provider": cls.LLM_PROVIDER,
            "llm_model": cls.LLM_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def validate_config(cls):
        """Validates that critical configurations are set."""
        if not cls.HF_API_TOKEN or "YOUR_TOKEN" in cls.HF_API_TOKEN:
            logger.error("HF_API_TOKEN is not set in the .env file.")
            return False
        if not cls.LLM_MODEL:
            logger.error("LLM_MODEL is not set in the .env file.")
            return False
        if not cls.LLM_PROVIDER:
            logger.error("LLM_PROVIDER is not set in the .env file.")
            return False
        logger.info("Configuration validated successfully.")
        return True

# Validate config on import
IS_CONFIG_VALID = Config.validate_config()