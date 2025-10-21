import streamlit as st
from config import Config
import pandas as pd
from logger import logger
import os

def show_settings():
    st.markdown("## ⚙️ System Settings & Configuration")
    
    try:
        stats = st.session_state.rag_engine.get_vector_store_stats()
    except Exception as e:
        logger.warning(f"Could not load vector store stats for settings page: {e}")
        stats = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # --- UPDATED: To show provider and new model var ---
        st.markdown(f"""
        <div class="feature-card">
            <h4>🤖 AI Models Status</h4>
            <p><strong>📊 Embedding Model:</strong> {Config.EMBEDDING_MODEL}</p>
            <p><strong>☁️ LLM Provider:</strong> {Config.LLM_PROVIDER}</p>
            <p><strong>🧠 Language Model:</strong> {Config.LLM_MODEL}</p>
            <p><strong>🔍 Vector Store:</strong> {stats.get('type', 'N/A')}</p>
            <p><strong>📈 Status:</strong> <span style="color: green;">✅ Operational</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        db_path = Config.CHROMA_DB_PATH if Config.VECTOR_STORE_TYPE == 'chroma' else Config.FAISS_DB_PATH
        st.markdown(f"""
        <div class="feature-card">
            <h4>💾 Data Storage</h4>
            <p><strong>📁 Vector DB Path:</strong> {db_path}</p>
            <p><strong>🔢 Embedding Dimension:</strong> {stats.get('dimension', 'N/A')}</p>
            <p><strong>📝 Text Chunking:</strong> {Config.CHUNK_SIZE} (Overlap: {Config.CHUNK_OVERLAP})</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documents", st.session_state.documents_added)
    with col2:
        st.metric("🌐 Total Web Pages", st.session_state.web_pages_crawled)
    with col3:
        st.metric("📝 Indexed Chunks", stats.get('index_size', 0))
    with col4:
        st.metric("💬 Chat Sessions", len(st.session_state.chat_history))
            
    st.markdown("### 🔧 System Information")
    system_info = {
        '🐍 Python Environment': "3.9+",
        '🌊 Streamlit Version': st.__version__,
        '🔍 Vector Store Engine': f"ChromaDB / FAISS (Config: {Config.VECTOR_STORE_TYPE})",
        '🕸️ Knowledge Graph Engine': 'NetworkX',
        '🤖 ML Framework': 'HuggingFace (API + Transformers)'
    }
    info_df = pd.DataFrame(list(system_info.items()), columns=['Component', 'Version/Status'])
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    st.markdown("### 💾 Data Management")
    st.info("Data management operations (like Load/Export) are now handled automatically by the persistent vector store (e.g., Chroma).")

    st.markdown("### 📜 Log Viewer")
    with st.expander("Click to view application logs (logs/rag_app.log)"):
        try:
            with open("logs/rag_app.log", "r") as f:
                logs = f.readlines()
                st.code("".join(logs[-50:]), language="log") # Show last 50 lines
        except FileNotFoundError:
            st.warning("Log file not found. It will be created when an event is logged.")