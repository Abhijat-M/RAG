import faiss
import numpy as np
import pickle
import os
import streamlit as st
import chromadb
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from logger import logger
import uuid

# --- ABSTRACT BASE CLASS ---

class BaseVectorStore(ABC):
    """Abstract interface for a vector store."""

    @abstractmethod
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        pass

    @st.cache_resource
    def _load_embedding_model(_self, model_name: str):
        """Loads and caches the sentence transformer model."""
        logger.info(f"Loading embedding model: {model_name}")
        try:
            model = SentenceTransformer(model_name)
            dimension = model.get_sentence_embedding_dimension()
            return model, dimension
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise

# --- CHROMA DB IMPLEMENTATION (Production Recommended) ---

class ChromaVectorStore(BaseVectorStore):
    """Vector database management using ChromaDB (Persistent)."""

    def __init__(self, path: str, model_name: str, collection_name="rag_collection"):
        self.path = path
        self.model_name = model_name
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        self.model, self.dimension = self._load_embedding_model(model_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        if not documents:
            return
            
        logger.info(f"Adding {len(documents)} documents to Chroma...")
        embeddings = self.model.encode(documents, convert_to_numpy=True).tolist()
        
        # Chroma needs unique IDs
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Add in batches to avoid overwhelming the DB
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadata[i:i+batch_size]
            batch_embeds = embeddings[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            try:
                self.collection.add(
                    embeddings=batch_embeds,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            except Exception as e:
                logger.error(f"Error adding batch to Chroma: {e}")
        logger.info("Document addition to Chroma complete.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
            
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count())
        )
        
        # Re-format Chroma's output to match the RAG engine's expected format
        formatted_results = []
        if not results.get('documents'):
            return []

        for i, doc in enumerate(results['documents'][0]):
            formatted_results.append({
                'document': doc,
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity score
            })
        return formatted_results

    def save(self):
        # Chroma is persistent, so save is a no-op.
        logger.info("ChromaVectorStore is persistent. No explicit save needed.")
        pass

    def load(self):
        # Chroma loads on init.
        logger.info("ChromaVectorStore loaded on initialization.")
        pass
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents. Warning: Can be memory-intensive."""
        count = self.collection.count()
        if count == 0:
            return []
        
        # Get all, this might be slow for >100k docs
        data = self.collection.get(
            limit=count, 
            include=["metadatas", "documents"]
        )
        
        return [
            {'content': doc, 'metadata': meta}
            for doc, meta in zip(data['documents'], data['metadatas'])
        ]

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_documents': self.collection.count(),
            'index_size': self.collection.count(),
            'dimension': self.dimension,
            'model': self.model_name,
            'type': 'Chroma'
        }

# --- FAISS IMPLEMENTATION (File-based) ---

class FAISSVectorStore(BaseVectorStore):
    """Vector database management using FAISS (File-based)."""

    def __init__(self, path: str, model_name: str):
        self.path = path
        self.index_file = os.path.join(path, "index.faiss")
        self.data_file = os.path.join(path, "documents.pkl")
        self.model_name = model_name
        self.model, self.dimension = self._load_embedding_model(model_name)
        self.index = None
        self.documents = []
        self.metadata = []

    def initialize_index(self):
        if self.index is None:
            logger.info("Initializing new FAISS index.")
            self.index = faiss.IndexFlatIP(self.dimension)

    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        if not documents:
            return
            
        self.initialize_index()
        logger.info(f"Adding {len(documents)} documents to FAISS...")
        
        embeddings = self.model.encode(documents, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        logger.info("Document addition to FAISS complete.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        return [
            {
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(score)
            }
            for score, idx in zip(scores[0], indices[0]) if idx >= 0
        ]

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        
        with open(self.data_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        logger.info(f"FAISSVectorStore saved to {self.path}")

    def load(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            logger.info(f"FAISSVectorStore loaded from {self.path}")
        else:
            logger.warning(f"No FAISS index found at {self.index_file}. Starting new.")
            self.initialize_index()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return [
            {'content': doc, 'metadata': meta}
            for doc, meta in zip(self.documents, self.metadata)
        ]

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'model': self.model_name,
            'type': 'FAISS'
        }