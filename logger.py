import logging
import sys
import os

def get_logger(name="RAG_App"):
    """
    Initializes and returns a centralized logger.
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/rag_app.log"), # Log to a file
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )
    logger = logging.getLogger(name)
    return logger

logger = get_logger()