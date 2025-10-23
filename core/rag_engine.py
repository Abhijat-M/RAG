
from __future__ import annotations

import os
import shutil
from typing import List, Dict, Any, Optional

import chromadb
from huggingface_hub import InferenceClient

from config import Config
from logger import logger
from storage import get_vector_store, BaseVectorStore


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    Uses a pluggable vector store for retrieval and HuggingFace
    InferenceClient (Featherless-AI) for generation.
    """

    def __init__(self) -> None:
        logger.info("Initializing RAGEngine…")

        # --- validation -------------------------------------------------
        if not Config.HF_API_TOKEN or not Config.LLM_MODEL:
            raise ValueError(
                "HF_API_TOKEN and LLM_MODEL must be set in environment or .env"
            )

        # --- vector store ----------------------------------------------
        self.vector_store: BaseVectorStore = get_vector_store()
        self.max_context_length: int = 1_500  # tokens

        # --- LLM client -------------------------------------------------
        self.llm_client = InferenceClient(
            provider=Config.LLM_PROVIDER,  # e.g. "featherless-ai"
            api_key=Config.HF_API_TOKEN,
        )
        logger.info(
            "RAGEngine ready – provider=%s model=%s",
            Config.LLM_PROVIDER,
            Config.LLM_MODEL,
        )

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index a list of documents."""
        if not documents:
            return
        contents = [doc["content"] for doc in documents]
        metas = [doc["metadata"] for doc in documents]
        self.vector_store.add_documents(contents, metas)
        self.vector_store.save()

    def retrieve_relevant_documents(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Fetch top-k relevant chunks."""
        return self.vector_store.search(query, k)

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Quick stats."""
        return self.vector_store.get_stats()

    def get_all_documents_for_kg(self) -> List[Dict[str, Any]]:
        """Return every indexed chunk (for Knowledge-Graph builders)."""
        logger.info("Retrieving all documents for KG build…")
        return self.vector_store.get_all_documents()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Low-level HF InferenceClient call."""
        logger.info("Calling HF API (%s messages)…", len(messages))
        try:
            completion = self.llm_client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                max_tokens=250,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM API error: %s", exc)
            if "overloaded" in str(exc).lower():
                return "Model is overloaded – please retry shortly."
            return f"LLM error: {exc}"

    def create_prompt(self, query: str, context: str) -> str:
        """Build a single-shot RAG prompt."""
        return (
            "You are a helpful assistant. Answer the question below **only** "
            "using the provided context. If the answer is absent, say "
            "\"I do not have that information in my documents.\"\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

    def prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """Concatenate chunks until max_context_length."""
        parts, total = [], 0
        for doc in docs:
            txt = doc.get("document", "")
            if total + len(txt) <= self.max_context_length:
                parts.append(txt)
                total += len(txt)
            else:
                rem = self.max_context_length - total
                if rem > 50:
                    parts.append(txt[:rem] + "…")
                break
        return "\n\n---\n\n".join(parts)

    def calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Average retrieval score clamped to [0,1]."""
        if not docs:
            return 0.0
        avg = sum(d.get("score", 0.0) for d in docs) / len(docs)
        return min(max(avg, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Single-turn RAG
    # ------------------------------------------------------------------
    def generate_response(
        self, query: str, context_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Standard RAG: retrieve → build prompt → generate."""
        if context_docs is None:
            context_docs = self.retrieve_relevant_documents(query, k=5)

        if not context_docs:
            logger.warning("No documents retrieved for query: %s", query)
            return {
                "answer": "I don't have enough information to answer your question.",
                "sources": [],
                "confidence": 0.0,
                "context_used": "",
            }

        context = self.prepare_context(context_docs)
        prompt = self.create_prompt(query, context)
        answer = self._generate_llm_response([{"role": "user", "content": prompt}])

        return {
            "answer": answer,
            "sources": [d["metadata"] for d in context_docs[:3]],
            "confidence": self.calculate_confidence(context_docs),
            "context_used": context[:500] + "…",
        }

    # ------------------------------------------------------------------
    # Multi-turn chat
    # ------------------------------------------------------------------
    def chat_mode(
        self, query: str, chat_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Context-aware conversational RAG."""
        # Build enhanced retrieval query from last 3 turns
        retrieval_parts = []
        for turn in chat_history[-3:]:
            retrieval_parts.append(f"Human: {turn['human']}\nAssistant: {turn['assistant']}")
        retrieval_parts.append(f"Human: {query}")
        enhanced_query = "\n".join(retrieval_parts)

        context_docs = self.retrieve_relevant_documents(enhanced_query, k=5)

        # Build message list for LLM
        messages = []
        for turn in chat_history[-3:]:
            messages.append({"role": "user", "content": turn["human"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        if context_docs:
            context = self.prepare_context(context_docs)
            prompt = self.create_prompt(query, context)
        else:
            logger.warning("No docs retrieved; answering without context.")
            prompt = query
        messages.append({"role": "user", "content": prompt})

        answer = self._generate_llm_response(messages)

        return {
            "answer": answer,
            "sources": [d["metadata"] for d in context_docs[:3]] if context_docs else [],
            "confidence": self.calculate_confidence(context_docs),
            "context_used": (context[:500] + "…") if context_docs else "No context found.",
        }

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------
    def clear_vector_store(self) -> None:
        """Wipe the vector DB and re-initialise."""
        logger.warning("Clearing vector store…")
        store_type = Config.VECTOR_STORE_TYPE

        # Drop singleton so next get_vector_store() creates a fresh instance
        import storage
        storage._vector_store_instance = None  # type: ignore

        if store_type == "chroma":
            try:
                client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
                client.reset()  # Chroma-native wipe
                logger.info("Chroma database reset.")
            except Exception as exc:
                logger.error("Chroma reset failed: %s", exc)
        elif store_type == "faiss" and os.path.exists(Config.FAISS_DB_PATH):
            shutil.rmtree(Config.FAISS_DB_PATH)
            logger.info("FAISS directory removed.")

        self.vector_store = get_vector_store()
        logger.info("Vector store cleared and re-initialised.")