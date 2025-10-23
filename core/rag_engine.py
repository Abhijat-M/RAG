import streamlit as st
from typing import List, Dict, Any, Optional
from storage import get_vector_store, BaseVectorStore
from config import Config
from logger import logger
from huggingface_hub import InferenceClient
from huggingface_hub.errors import TextGenerationError
import os
import chromadb
import shutil  # Keep shutil for FAISS
import time

class RAGEngine:
    """
    Core RAG (Retrieval Augmented Generation) engine.
    Uses an API for generation and a pluggable vector store.
    """
    
    def __init__(self):
        logger.info("Initializing RAGEngine...")
        self.vector_store: BaseVectorStore = get_vector_store()
        self.max_context_length = 1500  # Max tokens for context
        
        if not Config.HF_API_TOKEN or not Config.LLM_MODEL:
            msg = "Hugging Face API Token (HF_API_TOKEN) or Model (LLM_MODEL) not set."
            logger.error(msg)
            raise ValueError(msg)
            
        self.llm_client = InferenceClient(
            provider=Config.LLM_PROVIDER,
            api_key=Config.HF_API_TOKEN
        )
        logger.info(f"RAGEngine initialized with provider {Config.LLM_PROVIDER} and model {Config.LLM_MODEL}")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store."""
        if not documents:
            return
        
        contents = [doc['content'] for doc in documents]
        metadata = [doc['metadata'] for doc in documents]
        
        self.vector_store.add_documents(contents, metadata)
        self.vector_store.save() # Save after adding

    def retrieve_relevant_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        return self.vector_store.search(query, k)

    def _generate_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Calls the LLM API using the chat completions endpoint."""
        logger.info(f"Generating LLM chat completion for {len(messages)} messages...")
        try:
            completion = self.llm_client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                max_tokens=250, 
                temperature=0.7,
            )
            response_content = completion.choices[0].message.content
            logger.info("LLM response received.")
            return response_content.strip()
            
        except Exception as e:
            logger.error(f"LLM API Error: {e}")
            if "overloaded" in str(e):
                return "The AI model is currently overloaded. Please try again in a moment."
            return f"Error generating response from LLM: {e}"

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate response using RAG approach for a single query."""
        
        if context_docs is None:
            context_docs = self.retrieve_relevant_documents(query)
        
        if not context_docs:
            logger.warning(f"No documents found for query: {query}")
            return {
                'answer': "I don't have enough information to answer your question. Please add some documents first.",
                'sources': [],
                'confidence': 0.0
            }
        
        context = self.prepare_context(context_docs)
        prompt_string = self.create_prompt(query, context)
        
        messages = [
            {"role": "user", "content": prompt_string}
        ]
        
        answer = self._generate_llm_response(messages)
        
        return {
            'answer': answer,
            'sources': [doc['metadata'] for doc in context_docs[:3]],
            'confidence': self.calculate_confidence(context_docs),
            'context_used': context[:500] + "..."
        }
    
    def prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        total_length = 0
        
        for doc in docs:
            content = doc['document']
            content_len = len(content)
            
            if total_length + content_len <= self.max_context_length:
                context_parts.append(content)
                total_length += content_len
            else:
                remaining = self.max_context_length - total_length
                if remaining > 50:
                    context_parts.append(content[:remaining] + "...")
                break
        
        return "\n\n---\n\n".join(context_parts)

    def create_prompt(self, query: str, context: str) -> str:
        """Create prompt string for the language model."""
        return f"""You are a helpful assistant. Answer the following question based *only* on the provided context. If the answer is not in the context, say "I do not have that information in my documents."

Context:
{context}

Question:
{query}

Answer:"""
    
    def calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval scores."""
        if not docs:
            return 0.0
        avg_score = sum(doc.get('score', 0) for doc in docs) / len(docs)
        return min(max(avg_score, 0.0), 1.0)
    
    def get_all_documents_for_kg(self) -> List[Dict[str, Any]]:
        """Get all documents from the vector store for KG building."""
        logger.info("Retrieving all documents for KG build...")
        return self.vector_store.get_all_documents()

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_store.get_stats()
    
    # --- THIS FUNCTION IS NOW FIXED ---
    def clear_vector_store(self):
        """Clear all documents from vector store."""
        logger.warning("Clearing vector store...")
        store_type = Config.VECTOR_STORE_TYPE
        
        # This will force re-initialization on next get()
        from storage import _vector_store_instance
        _vector_store_instance = None 
        
        if store_type == 'chroma':
            try:
                # --- FIX: Use client.reset() for a clean reset ---
                # This properly clears the DB and the client's internal state.
                client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
                client.reset() # This deletes all collections
                logger.info(f"Chroma database at {Config.CHROMA_DB_PATH} reset.")
                # --- END FIX ---
            except Exception as e:
                logger.error(f"Error resetting chroma database: {e}")
        elif store_type == 'faiss':
            if os.path.exists(Config.FAISS_DB_PATH):
                shutil.rmtree(Config.FAISS_DB_PATH)
                logger.info(f"FAISS directory {Config.FAISS_DB_PATH} deleted.")
        
        # Re-create the vector_store instance for the current engine object
        self.vector_store = get_vector_store() 
        logger.info("Vector store cleared and re-initialized.")
    
    def chat_mode(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Enhanced chat mode with conversation history."""
        
        # 1. Build message history for retrieval
        retrieval_query_parts = []
        if chat_history:
            for exchange in chat_history[-3:]: # Use last 3 exchanges
                retrieval_query_parts.append(f"Human: {exchange['human']}\nAssistant: {exchange['assistant']}")
        retrieval_query_parts.append(f"Human: {query}")
        enhanced_query = "\n".join(retrieval_query_parts)
        
        # 2. Retrieve docs
        context_docs = self.retrieve_relevant_documents(enhanced_query)
        
        # 3. Build message history for generation
        messages = []
        if chat_history:
            for exchange in chat_history[-3:]: # Use last 3 exchanges
                messages.append({"role": "user", "content": exchange['human']})
                messages.append({"role": "assistant", "content": exchange['assistant']})
        
        # 4. Inject context into the *last* user message
        if context_docs:
            context = self.prepare_context(context_docs)
            prompt_with_context = self.create_prompt(query, context)
            messages.append({"role": "user", "content": prompt_with_context})
        else:
            logger.warning(f"No documents found for chat query: {query}. Answering without context.")
            messages.append({"role": "user", "content": query})

        # 5. Generate response
        answer = self._generate_llm_response(messages)
        
        return {
            'answer': answer,
            'sources': [doc['metadata'] for doc in context_docs[:3]], 
            'confidence': self.calculate_confidence(context_docs),
            'context_used': context[:500] + "..." if context_docs else "No context found."
        }