import streamlit as st

def show_dashboard():
    st.markdown("## 📊 System Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        stats = st.session_state.rag_engine.get_vector_store_stats()
    except:
        stats = {}
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📄</h3>
            <h2>{st.session_state.documents_added}</h2>
            <p>Documents Uploaded</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌐</h3>
            <h2>{st.session_state.web_pages_crawled}</h2>
            <p>Web Pages Crawled</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        chunks = stats.get('total_documents', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔍</h3>
            <h2>{chunks}</h2>
            <p>Text Chunks Indexed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        chats = len(st.session_state.chat_history)
        st.markdown(f"""
        <div class="metric-card">
            <h3>💬</h3>
            <h2>{chats}</h2>
            <p>Chat Interactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 System Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📄 Document Processing</h4>
            <p>Upload and process PDF, DOCX, TXT, and MD files. Processing runs in the background.</p>
            <ul>
                <li>✅ Multi-format support</li>
                <li>✅ Asynchronous ingestion</li>
                <li>✅ Vector embeddings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🕸️ Knowledge Graphs</h4>
            <p>Automatically extract entities and relationships to build interactive knowledge graphs.</p>
            <ul>
                <li>✅ Entity extraction</li>
                <li>✅ Background processing</li>
                <li>✅ Interactive visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🌐 Web Crawling</h4>
            <p>Extract information from websites with depth limits and context-aware filtering.</p>
            <ul>
                <li>✅ Recursive crawling</li>
                <li>✅ Depth & page limits</li>
                <li>✅ Context filtering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 AI Chat Interface</h4>
            <p>Intelligent Q&A system powered by a scalable RAG architecture using an external API.</p>
            <ul>
                <li>✅ Scalable API-based LLM</li>
                <li>✅ Source attribution</li>
                <li>✅ Conversation history</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        st.markdown("### 📈 Recent Chat Activity")
        recent_chats = st.session_state.chat_history[-5:]
        for i, chat in enumerate(reversed(recent_chats)):
            with st.expander(f"💬 Query {len(st.session_state.chat_history) - i}: {chat['human'][:50]}..."):
                st.markdown(f"**Question:** {chat['human']}")
                st.markdown(f"**Answer:** {chat['assistant']}")
                if 'confidence' in chat:
                    st.progress(chat['confidence'])
                    st.caption(f"Confidence: {chat['confidence']:.2%}")