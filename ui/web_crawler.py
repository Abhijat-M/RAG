import streamlit as st
from logger import logger

def show_web_crawler(crawl_urls_background_fn):
    st.markdown("## 🌐 Web Crawler & Information Extraction")
    
    st.markdown("""
    <div class="feature-card">
        <h4>🔍 Intelligent Web Crawling</h4>
        <p>Extract valuable information from websites. Set limits for depth and total pages per site, and provide context keywords (comma-separated) to filter for relevant content.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        urls_input = st.text_area(
            "🔗 Enter Root URLs (one per line)",
            placeholder="https://example.com/blog\nhttps://another-site.com/news\nhttps://docs.example.org",
            height=120,
            help="Add multiple root URLs to crawl simultaneously",
            key="urls_input"
        )
    
    with col2:
        st.markdown("### ⚙️ Crawl Settings")
        max_pages = st.slider("📊 Max Pages per URL", 1, 50, 5, key="max_pages")
        max_depth = st.slider("🕸️ Max Depth per URL", 0, 5, 2, key="max_depth", help="0 = only the starting page, 1 = starting page + its links, etc.")
        context = st.text_input(
            "🎯 Context Keywords (optional)", 
            placeholder="AI, machine learning, data science",
            help="Comma-separated. Filters pages to only include those matching keywords.",
            key="context_input"
        )
    
    if urls_input.strip():
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        st.markdown(f"### 🎯 Ready to Crawl {len(urls)} URL(s)")
        
        add_to_kb = st.checkbox(
            "✅ Add crawled content to Knowledge Base", 
            value=True, 
            key="add_to_vector_store_toggle",
            help="If checked, the extracted text will be added to the vector store for RAG."
        )
        
        if st.button("🚀 Start Crawling", type="primary", key="start_crawl"):
            pool = st.session_state.process_pool
            future = pool.submit(
                crawl_urls_background_fn,
                urls,
                context,
                max_pages,
                max_depth,
                add_to_kb  # Pass the new option
            )
            st.session_state.jobs['crawling'] = future
            st.info("🕷️ Crawling websites in the background. This may take a while...")

    # Check for job completion
    if 'crawling' in st.session_state.jobs:
        future = st.session_state.jobs['crawling']
        if future.running():
            st.spinner("🕷️ Crawling websites in the background...")
        elif future.done():
            try:
                pages_crawled, pages_added = future.result()  # Unpack both values
                
                # Create a dynamic message
                if pages_added > 0:
                    success_msg = f"Successfully crawled <strong>{pages_crawled}</strong> pages and added <strong>{pages_added}</strong> to the knowledge base."
                    st.session_state.web_pages_crawled += pages_added # Only increment by what was added
                else:
                    success_msg = f"Successfully crawled <strong>{pages_crawled}</strong> pages. (Not added to knowledge base)"

                st.markdown(f"""
                <div class="success-message">
                    <h4>✅ Crawling Complete!</h4>
                    <p>{success_msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                del st.session_state.jobs['crawling']
            except Exception as e:
                logger.error(f"Crawling job failed: {e}")
                st.error(f"Crawling job failed: {e}")
                del st.session_state.jobs['crawling']