import streamlit as st
from logger import logger

def show_chat_interface():
    st.markdown("## 💬 AI-Powered Chat Interface")
    rag_engine = st.session_state.rag_engine
    
    # Check if we have any data
    try:
        stats = rag_engine.get_vector_store_stats()
        has_data = stats.get('total_documents', 0) > 0
    except:
        has_data = False
    
    if not has_data:
        st.markdown("""
        <div class="warning-message">
            <h4>📋 No Knowledge Base Found</h4>
            <p>Please upload documents or crawl websites first to enable the chat functionality.</p>
            <p>👉 Go to <strong>Document Upload</strong> or <strong>Web Crawler</strong> to add content.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Chat history display
    st.markdown("### 🗨️ Conversation History")
    chat_container = st.container(height=400)
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="feature-card">
                <h4>👋 Welcome to RAG 2.0 Chat!</h4>
                <p>Ask me anything about your uploaded documents or crawled web content. I'll provide intelligent answers with source references.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, chat in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 You:</strong><br>
                    {chat['human']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {chat['assistant']}
                </div>
                """, unsafe_allow_html=True)
                
                if 'confidence' in chat and chat['confidence'] > 0:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.progress(chat['confidence'])
                        st.caption(f"Confidence: {chat['confidence']:.1%}")
                    with col2:
                        if 'sources' in chat and chat['sources']:
                            with st.expander(f"📚 Sources ({len(chat['sources'])})"):
                                for j, source in enumerate(chat['sources']):
                                    filename = source.get('filename', source.get('title', source.get('url', 'Unknown')))
                                    st.caption(f"📄 {filename}")
    
    # Chat input section
    st.markdown("### ✍️ Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "💭 Your question:",
            placeholder="What are the main topics discussed in the documents?",
            key="chat_input",
            label_visibility="collapsed",
            help="Ask anything about your knowledge base"
        )
    
    with col2:
        send_button = st.button("📤 Send", type="primary", key="send_chat", use_container_width=True)
    
    with st.expander("🔧 Advanced Chat Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_sources = st.slider("📊 Sources to retrieve (k)", 1, 10, 5, key="num_sources")
        with col2:
            chat_mode = st.checkbox("🧠 Use conversation context", value=True, key="chat_mode")
    
    if send_button and user_question.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                if chat_mode:
                    response = rag_engine.chat_mode(
                        user_question, 
                        st.session_state.chat_history
                    )
                else:
                    context_docs = rag_engine.retrieve_relevant_documents(user_question, k=num_sources)
                    response = rag_engine.generate_response(user_question, context_docs=context_docs)
                
                st.session_state.chat_history.append({
                    'human': user_question,
                    'assistant': response['answer'],
                    'sources': response.get('sources', []),
                    'confidence': response.get('confidence', 0.0),
                })
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error in chat response: {e}")
                st.error(f"An error occurred: {e}")