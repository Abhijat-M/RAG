import networkx as nx
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict
import plotly.graph_objects as go
from logger import logger

class KnowledgeGraphBuilder:
    """Build and visualize knowledge graphs from documents"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entities = set()
        self.relationships = []
        self.entity_types = {}
        
    def extract_entities_and_relationships(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract entities and relationships from documents"""
        logger.info(f"Starting KG extraction from {len(documents)} documents...")
        all_entities = set()
        all_relationships = []
        
        # Limit processing to first 1000 documents to avoid crashing
        docs_to_process = documents[:1000]
        
        for i, doc in enumerate(docs_to_process):
            if i % 100 == 0:
                logger.info(f"Processing document {i}/{len(docs_to_process)} for KG...")
            content = doc.get('content', '')
            if not content:
                continue
                
            entities = self.extract_entities_simple(content)
            relationships = self.extract_relationships_simple(content, entities)
            
            all_entities.update(entities)
            all_relationships.extend(relationships)
        
        self.build_graph(all_entities, all_relationships)
        logger.info(f"KG build complete. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        
        return {
            'entities_count': len(all_entities),
            'relationships_count': len(all_relationships),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges()
        }
    
    def extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction using regex patterns"""
        entities = []
        
        # Capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(proper_nouns)
        
        # Organizations (words ending with common org suffixes)
        orgs = re.findall(r'\b\w+(?:\s+\w+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Organization|Institute|University|College))\b', text, re.IGNORECASE)
        entities.extend(orgs)
        
        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', text)
        entities.extend(dates)
        
        # Remove duplicates and filter
        entities = list(set([e.strip() for e in entities if len(e.strip()) > 2]))
        return entities[:100]  # Limit per document
    
    def extract_relationships_simple(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Extract simple relationships between entities"""
        relationships = []
        entity_set = set(entities)
        
        patterns = [
            (r'(\w+(?:\s+\w+)*)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)', 'is_a'),
            (r'(\w+(?:\s+\w+)*)\s+(?:works for|employed by|part of)\s+(\w+(?:\s+\w+)*)', 'works_for'),
            (r'(\w+(?:\s+\w+)*)\s+(?:located in|based in|from)\s+(\w+(?:\s+\w+)*)', 'located_in'),
            (r'(\w+(?:\s+\w+)*)\s+(?:created|founded|established)\s+(\w+(?:\s+\w+)*)', 'created'),
            (r'(\w+(?:\s+\w+)*)\s+(?:and|with|alongside)\s+(\w+(?:\s+\w+)*)', 'associated_with')
        ]
        
        for pattern, rel_type in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        entity1, entity2 = match[0].strip(), match[1].strip()
                        if entity1 in entity_set and entity2 in entity_set and entity1 != entity2:
                            relationships.append((entity1, entity2, rel_type))
            except re.error as e:
                logger.warning(f"Regex error in KG extraction: {e}")
                pass
        
        return relationships
    
    def build_graph(self, entities: List[str], relationships: List[Tuple[str, str, str]]):
        """Build NetworkX graph from entities and relationships"""
        self.graph.clear()
        
        for entity in entities:
            self.graph.add_node(entity, type=self.classify_entity(entity))
        
        for entity1, entity2, rel_type in relationships:
            if self.graph.has_node(entity1) and self.graph.has_node(entity2):
                self.graph.add_edge(entity1, entity2, relationship=rel_type)
    
    def classify_entity(self, entity: str) -> str:
        """Simple entity classification"""
        entity_lower = entity.lower()
        
        if re.search(r'\d{4}', entity) or any(month in entity_lower for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
            return 'date'
        elif any(word in entity_lower for word in ['inc', 'corp', 'llc', 'ltd', 'company', 'organization']):
            return 'organization'
        elif any(word in entity_lower for word in ['university', 'college', 'school', 'institute']):
            return 'institution'
        elif entity[0].isupper() and ' ' in entity and len(entity.split()) <= 3:
            return 'person'
        elif entity[0].isupper() and len(entity.split()) == 1:
            return 'concept'
        else:
            return 'concept'
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if self.graph.number_of_nodes() == 0:
            return {'message': 'No graph data available'}
        
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }
        
        if stats['nodes'] > 1:
            try:
                degree_centrality = nx.degree_centrality(self.graph)
                stats['most_central_nodes'] = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                
                if subgraph.number_of_nodes() > 1:
                    stats['diameter'] = nx.diameter(subgraph)
                    stats['average_path_length'] = nx.average_shortest_path_length(subgraph)
            except Exception as e:
                logger.warning(f"Could not calculate complex graph metrics: {e}")
                pass
        
        return stats
    
    def visualize_graph_plotly(self) -> go.Figure:
        """Create interactive graph visualization using Plotly"""
        if self.graph.number_of_nodes() == 0:
            fig = go.Figure()
            fig.add_annotation(text="No graph data to display", 
                             xref="paper", yref="paper", 
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)
        
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        color_map = {
            'person': 'lightblue', 'organization': 'lightgreen',
            'institution': 'orange', 'date': 'pink', 'concept': 'lightgray'
        }
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_type = self.graph.nodes[node].get('type', 'concept')
            node_info = f"{node}<br>Type: {node_type}"
            node_text.append(node_info)
            node_color.append(color_map.get(node_type, 'lightgray'))
            degree = self.graph.degree[node]
            node_size.append(max(10, min(degree * 5, 50)))
        
        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none', mode='lines', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in self.graph.nodes()],
            hovertext=node_text,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color='black')),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Knowledge Graph Visualization",
            showlegend=False, hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Drag to pan, scroll to zoom",
                showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig
    
    def find_shortest_path(self, entity1: str, entity2: str) -> List[str]:
        """Find shortest path between two entities"""
        try:
            if entity1 in self.graph.nodes() and entity2 in self.graph.nodes():
                return nx.shortest_path(self.graph, entity1, entity2)
        except nx.NetworkXNoPath:
            return []
        return []
    
    def get_entity_neighbors(self, entity: str, depth: int = 1) -> List[str]:
        """Get neighboring entities within specified depth"""
        if entity not in self.graph.nodes():
            return []
        
        try:
            neighbor_nodes = nx.ego_graph(self.graph, entity, radius=depth).nodes()
            neighbors = list(neighbor_nodes)
            neighbors.remove(entity)
            return neighbors
        except Exception:
            return list(self.graph.neighbors(entity))