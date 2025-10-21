from config import Config
from logger import logger
from .vector_store import BaseVectorStore, FAISSVectorStore, ChromaVectorStore

# Singleton instance
_vector_store_instance = None

def get_vector_store() -> BaseVectorStore:
    """
    Factory function to get the singleton instance of the configured vector store.
    """
    global _vector_store_instance
    
    if _vector_store_instance is not None:
        return _vector_store_instance

    if Config.VECTOR_STORE_TYPE.lower() == 'chroma':
        logger.info(f"Initializing ChromaVectorStore at {Config.CHROMA_DB_PATH}")
        _vector_store_instance = ChromaVectorStore(
            path=Config.CHROMA_DB_PATH,
            model_name=Config.EMBEDDING_MODEL
        )
    elif Config.VECTOR_STORE_TYPE.lower() == 'faiss':
        logger.info(f"Initializing FAISSVectorStore at {Config.FAISS_DB_PATH}")
        _vector_store_instance = FAISSVectorStore(
            path=Config.FAISS_DB_PATH,
            model_name=Config.EMBEDDING_MODEL
        )
    else:
        logger.error(f"Unknown VECTOR_STORE_TYPE: {Config.VECTOR_STORE_TYPE}")
        raise ValueError(f"Unknown VECTOR_STORE_TYPE: {Config.VECTOR_STORE_TYPE}")
    
    try:
        _vector_store_instance.load()
        logger.info("Vector store loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load vector store (may be new): {e}")

    return _vector_store_instance