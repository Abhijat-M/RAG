import streamlit as st
from config import Config
from logger import logger
import pandas as pd

def show_document_upload(process_files_background_fn):
    st.markdown("## 📄 Document Upload & Processing")
    
    st.markdown(f"""
    <div class="feature-card">
        <h4>📋 Supported Document Formats</h4>
        <p>Upload your documents in any of the following formats:</p>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            {' '.join([f'<span style="background: #e3f2fd; padding: 5px 10px; border-radius: 15px;">{ext}</span>' for ext in Config.ALLOWED_EXTENSIONS])}
        </div>
        <p style="margin-top: 10px;"><strong>Maximum file size:</strong> {Config.MAX_FILE_SIZE}MB per file</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📁 Choose files to upload",
        type=[ext.strip('.') for ext in Config.ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
        help="Upload documents to build your knowledge base"
    )
    
    if uploaded_files:
        st.markdown(f"### 📁 Ready to Process {len(uploaded_files)} File(s)")
        
        file_details = []
        for file in uploaded_files:
            file_details.append({"Name": file.name, "Size (bytes)": file.size, "Type": file.type})
        st.dataframe(pd.DataFrame(file_details), use_container_width=True)
        
        if st.button("🚀 Process Documents", type="primary", key="process_docs"):
            pool = st.session_state.process_pool
            
            # Submit background job
            future = pool.submit(process_files_background_fn, uploaded_files)
            st.session_state.jobs['file_processing'] = future
            st.info("🔄 Processing documents in the background. You can navigate away or check the logs.")

    # Check for job completion
    if 'file_processing' in st.session_state.jobs:
        future = st.session_state.jobs['file_processing']
        if future.running():
            st.spinner("🔄 Processing documents in the background...")
        elif future.done():
            try:
                files, chunks = future.result()
                st.markdown(f"""
                <div class="success-message">
                    <h4>✅ Processing Complete!</h4>
                    <p>Successfully processed <strong>{chunks}</strong> chunks from <strong>{files}</strong> files.</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.documents_added += files
                del st.session_state.jobs['file_processing']
            except Exception as e:
                logger.error(f"File processing job failed: {e}")
                st.error(f"File processing job failed: {e}")
                del st.session_state.jobs['file_processing']