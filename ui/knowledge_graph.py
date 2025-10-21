import streamlit as st
from logger import logger

def show_knowledge_graph(build_kg_background_fn):
    st.markdown("## 🕸️ Knowledge Graph Visualization")
    
    st.markdown("""
    <div class="feature-card">
        <h4>🧠 Intelligent Knowledge Mapping</h4>
        <p>Explore the relationships between entities automatically extracted from your documents and web content. 
        Click "Build Knowledge Graph" to process <strong>all</strong> content in the vector store.</p>
        <p><i>Note: This will run in the background and may take a moment for large knowledge bases.</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        stats = st.session_state.rag_engine.get_vector_store_stats()
        has_data = stats.get('total_documents', 0) > 0
    except:
        has_data = False
    
    if not has_data:
        st.markdown("""
        <div class="warning-message">
            <h4>📊 No Data Available</h4>
            <p>Upload documents or crawl websites first to generate knowledge graphs.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### ⚙️ Graph Controls")
        
        if st.button("🔄 Build/Update Knowledge Graph", key="build_graph", type="primary", use_container_width=True):
            pool = st.session_state.process_pool
            future = pool.submit(build_kg_background_fn)
            st.session_state.jobs['kg_build'] = future
            st.info("🧠 Building Knowledge Graph in the background...")

        # Graph statistics
        try:
            graph_stats = st.session_state.kg_builder.get_graph_statistics()
            if 'nodes' in graph_stats and graph_stats['nodes'] > 0:
                st.metric("🎯 Entities", graph_stats['nodes'])
                st.metric("🔗 Relationships", graph_stats['edges'])
                st.metric("📊 Density", f"{graph_stats.get('density', 0):.3f}")
            else:
                st.info("📊 Build the graph to see statistics")
        except Exception as e:
            st.warning(f"Stats unavailable: {e}")

    # Check for job completion
    if 'kg_build' in st.session_state.jobs:
        future = st.session_state.jobs['kg_build']
        if future.running():
            st.spinner("🧠 Building Knowledge Graph in the background...")
        elif future.done():
            try:
                kg_builder_instance, stats = future.result()
                if kg_builder_instance:
                    st.session_state.kg_builder = kg_builder_instance # Update session state
                    st.markdown(f"""
                    <div class="success-message">
                        <h4>✅ Knowledge Graph Built!</h4>
                        <p>Found {stats.get('graph_nodes', 0)} entities and {stats.get('graph_edges', 0)} relationships.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No documents found to build graph.")
                del st.session_state.jobs['kg_build']
                st.rerun()
            except Exception as e:
                logger.error(f"KG build job failed: {e}")
                st.error(f"KG build job failed: {e}")
                del st.session_state.jobs['kg_build']

    with col1:
        # Graph visualization
        try:
            if st.session_state.kg_builder.graph.number_of_nodes() > 0:
                st.markdown("### 🎨 Interactive Knowledge Graph")
                
                fig = st.session_state.kg_builder.visualize_graph_plotly()
                st.plotly_chart(fig, use_container_width=True, height=600)
                
                st.markdown("### 🔍 Entity Explorer")
                entities = sorted(list(st.session_state.kg_builder.graph.nodes()))
                
                if entities:
                    selected_entity = st.selectbox("🎯 Select entity to explore:", entities, key="entity_select")
                    
                    if selected_entity:
                        neighbors = st.session_state.kg_builder.get_entity_neighbors(selected_entity)
                        
                        col1_inner, col2_inner = st.columns(2)
                        
                        with col1_inner:
                            st.markdown(f"**🔗 Connected Entities ({len(neighbors)}):**")
                            st.dataframe(neighbors, use_container_width=True, height=150)
                        
                        with col2_inner:
                            if len(entities) > 1:
                                other_entities = [e for e in entities if e != selected_entity]
                                target_entity = st.selectbox("🎯 Find path to:", other_entities, key="target_select")
                                
                                if target_entity:
                                    path = st.session_state.kg_builder.find_shortest_path(selected_entity, target_entity)
                                    if path:
                                        st.markdown(f"**🛤️ Shortest path ({len(path)-1} steps):**")
                                        st.info(" → ".join(path))
                                    else:
                                        st.warning("❌ No path found")
            else:
                st.markdown("""
                <div class="feature-card">
                    <h4>🎯 Ready to Build Your Knowledge Graph</h4>
                    <p>Click "Build/Update Knowledge Graph" to automatically extract entities and relationships from all your content.</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error displaying graph: {e}")