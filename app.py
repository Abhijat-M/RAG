import streamlit as st
import traceback
from concurrent.futures import ProcessPoolExecutor
from config import Config, IS_CONFIG_VALID
from logger import logger
from core.rag_engine import RAGEngine
from core.knowledge_graph import KnowledgeGraphBuilder

# --- Import UI Pages ---
from ui import dashboard, document_upload, web_crawler, chat_interface, knowledge_graph, settings

# Set page config first
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Background Task Definitions ---
# (These must be top-level functions for ProcessPoolExecutor)

def process_files_background(uploaded_files):
    """Background task for processing files."""
    from ingestion.document_processor import DocumentProcessor
    from storage import get_vector_store
    
    logger.info(f"BG_TASK: Starting processing for {len(uploaded_files)} files.")
    processor = DocumentProcessor()
    # Read files into memory here, as they can't be passed directly
    file_contents = []
    for file in uploaded_files:
        file_contents.append({
            'name': file.name,
            'type': file.type,
            'data': file.getvalue()
        })

    processed_docs = processor.process_uploaded_files(file_contents)
    
    if processed_docs:
        rag_engine = RAGEngine()
        rag_engine.add_documents(processed_docs)
        logger.info(f"BG_TASK: File processing complete. Added {len(processed_docs)} chunks.")
        return len(uploaded_files), len(processed_docs)
    return 0, 0

def crawl_urls_background(urls: list[str], context: str, max_pages: int, max_depth: int, add_to_kb: bool):
    """Background task for crawling URLs."""
    from ingestion.web_crawler import WebCrawler
    
    logger.info(f"BG_TASK: Starting crawl for {len(urls)} URLs. Add to KB: {add_to_kb}")
    crawler = WebCrawler()
    crawled_content = crawler.crawl_root_urls(urls, context, max_pages, max_depth)
    
    pages_crawled = len(crawled_content)
    pages_added = 0

    if crawled_content and add_to_kb:
        # Re-init RAGEngine in the child process
        rag_engine = RAGEngine() 
        rag_engine.add_documents(crawled_content)
        pages_added = len(crawled_content)
        logger.info(f"BG_TASK: Crawling complete. Added {pages_added} pages to KB.")
    elif crawled_content:
        logger.info(f"BG_TASK: Crawling complete. Found {pages_crawled} pages (NOT adding to KB).")
    else:
        logger.info("BG_TASK: Crawling complete. No content found.")

    return pages_crawled, pages_added # Return both stats

def build_kg_background():
    """Background task for building the knowledge graph."""
    logger.info("BG_TASK: Starting Knowledge Graph build.")
    # Re-init RAGEngine in the child process
    rag_engine = RAGEngine() 
    all_docs = rag_engine.get_all_documents_for_kg()
    
    if not all_docs:
        logger.warning("BG_TASK: No documents found for KG build.")
        return None, {}
        
    kg_builder = KnowledgeGraphBuilder()
    stats = kg_builder.extract_entities_and_relationships(all_docs)
    logger.info(f"BG_TASK: KG build complete. Found {stats.get('graph_nodes')} nodes.")
    # Return the builder instance and stats
    return kg_builder, stats

# --- Main Application ---

# --- NEW DARK MODE CSS ---
st.markdown("""
<style>
    /* Base theme */
    body {
        color: #FAFAFA;
        background-color: #0E1117;
    }
    .main {
        background-color: #0E1117;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B0764 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: #1B1F2A; /* Slightly lighter dark */
        color: #FAFAFA;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #1B1F2A 0%, #2A2F3A 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3B82F6; /* Blue accent */
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        color: #FAFAFA; /* Ensure text is light */
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
    }
    .feature-card h4 {
        color: #3B82F6; /* Blue accent for headers */
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B0764 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s ease-in;
        color: #FAFAFA;
    }
    .user-message {
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
        border-left: 4px solid #93C5FD;
    }
    .assistant-message {
        background: linear-gradient(135deg, #374151 0%, #1F2937 100%);
        border-left: 4px solid #9CA3AF;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #1D4ED8 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }

    /* Input boxes */
    .stTextInput, .stTextArea {
        background-color: #1B1F2A;
        border-radius: 10px;
    }
    
    /* Success/Warning Messages */
    .success-message {
        background: linear-gradient(135deg, #064E3B 0%, #047857 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #34D399;
        margin: 1rem 0;
        color: white;
    }
    .warning-message {
        background: linear-gradient(135deg, #78350F 0%, #B45309 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
        color: white;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)
# --- END NEW CSS ---


@st.cache_resource
def get_rag_engine():
    """Cache the RAG engine instance."""
    return RAGEngine()

def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = get_rag_engine()
    if 'kg_builder' not in st.session_state:
        st.session_state.kg_builder = KnowledgeGraphBuilder()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_added' not in st.session_state:
        st.session_state.documents_added = 0
    if 'web_pages_crawled' not in st.session_state:
        st.session_state.web_pages_crawled = 0
    
    # Process pool for background tasks
    if 'process_pool' not in st.session_state:
        st.session_state.process_pool = ProcessPoolExecutor(max_workers=2)
    
    # To track background job futures
    if 'jobs' not in st.session_state:
        st.session_state.jobs = {}

def main():
    if not IS_CONFIG_VALID:
        st.error(
            "CRITICAL ERROR: Configuration is invalid. "
            "Please check your .env file for HF_API_TOKEN and LLM_MODEL_API."
        )
        st.stop()
        
    initialize_session_state()
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{Config.PAGE_ICON} {Config.PAGE_TITLE}</h1>
        <p>🚀 Intelligent Document Analysis • 🌐 Web Crawling • 🕸️ Knowledge Graphs • 🤖 AI-Powered Q&A</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🚀 Navigation")
        page = st.selectbox(
            "Choose a feature:",
            ["📊 Dashboard", "📄 Document Upload", "🌐 Web Crawler", "💬 Chat Interface", "🕸️ Knowledge Graph", "⚙️ Settings"],
            key="navigation"
        )
        
        st.markdown("### 📈 System Stats")
        try:
            stats = st.session_state.rag_engine.get_vector_store_stats()
        except Exception as e:
            logger.error(f"Could not get vector stats: {e}")
            stats = {}
        
        col1, col2 = st.columns(2)
        col1.metric("📄 Documents", st.session_state.documents_added)
        col2.metric("🌐 Web Pages", st.session_state.web_pages_crawled)
        st.metric("📝 Text Chunks", stats.get('total_documents', 0))
        
        st.markdown("### ⚡ Quick Actions")
        if st.button("🗑️ Clear All Data", key="clear_data"):
            try:
                st.session_state.rag_engine.clear_vector_store()
                st.session_state.kg_builder = KnowledgeGraphBuilder()
                st.session_state.chat_history = []
                st.session_state.documents_added = 0
                st.session_state.web_pages_crawled = 0
                st.success("✅ All data cleared!")
                st.rerun()
            except Exception as e:
                logger.error(f"Error clearing data: {e}")
                st.error(f"Error clearing data: {e}")
    
    # Page routing
    try:
        if page == "📊 Dashboard":
            dashboard.show_dashboard()
        elif page == "📄 Document Upload":
            document_upload.show_document_upload(process_files_background)
        elif page == "🌐 Web Crawler":
            web_crawler.show_web_crawler(crawl_urls_background)
        elif page == "💬 Chat Interface":
            chat_interface.show_chat_interface()
        elif page == "🕸️ Knowledge Graph":
            knowledge_graph.show_knowledge_graph(build_kg_background)
        elif page == "⚙️ Settings":
            settings.show_settings()
    except Exception as e:
        logger.error(f"Error loading page {page}: {e}")
        st.error(f"Error loading page: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()