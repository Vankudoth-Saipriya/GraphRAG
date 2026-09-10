import os
import json
import logging
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional, Set
from src.models import GraphTriplet, EntityNode, SubGraphContext
from src.config import settings

logger = logging.getLogger(__name__)

class GraphEngine:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or settings.graph_storage_path
        self.graph = nx.DiGraph()
        self._load_graph()

    def add_triplets(self, triplets: List[GraphTriplet]):
        """Add triplets (Subject -[Relation]-> Object) to the NetworkX graph."""
        for t in triplets:
            subj = t.subject.strip()
            obj = t.object.strip()
            rel = t.relation.strip()
            
            if not self.graph.has_node(subj):
                self.graph.add_node(subj, type="Entity", label=subj)
            if not self.graph.has_node(obj):
                self.graph.add_node(obj, type="Entity", label=obj)
                
            self.graph.add_edge(
                subj, 
                obj, 
                relation=rel, 
                confidence=t.confidence,
                source_doc_id=t.source_doc_id or ""
            )
        self.save_graph()

    def find_matching_entities(self, query_text: str, score_threshold: float = 65.0) -> List[str]:
        """Find entity nodes in graph that match terms in the query text using exact & RapidFuzz fuzzy matching."""
        if not self.graph.nodes():
            return []
            
        matched = set()
        query_text_lower = query_text.lower()
        nodes_list = [str(n) for n in self.graph.nodes()]

        # 1. Direct substring matching
        for node in nodes_list:
            node_lower = node.lower()
            if node_lower in query_text_lower or any(word in node_lower for word in query_text_lower.split() if len(word) > 3):
                matched.add(node)

        # 2. RapidFuzz similarity matching
        try:
            from rapidfuzz import process, fuzz
            # Extract top fuzzy matches for query terms
            for token in query_text.split():
                if len(token) > 2:
                    results = process.extract(token, nodes_list, scorer=fuzz.WRatio, score_cutoff=score_threshold, limit=3)
                    for match, score, _ in results:
                        matched.add(match)
        except Exception as e:
            logger.debug(f"RapidFuzz matching fallback: {e}")

        return list(matched)

    def get_subgraph(self, target_entity: str, hops: int = 2) -> SubGraphContext:
        """Extract N-hop neighborhood subgraph around target entity."""
        if not self.graph.has_node(target_entity):
            # Try case-insensitive lookup
            for n in self.graph.nodes():
                if str(n).lower() == target_entity.lower():
                    target_entity = n
                    break
            else:
                return SubGraphContext(target_entity=target_entity, triplets=[], neighbor_nodes=[])

        # Extract ego network (subgraph within N hops)
        sub_nodes = set([target_entity])
        current_layer = set([target_entity])
        
        for _ in range(hops):
            next_layer = set()
            for node in current_layer:
                # Predecessors and successors
                neighbors = set(self.graph.predecessors(node)).union(set(self.graph.successors(node)))
                next_layer.update(neighbors)
            sub_nodes.update(next_layer)
            current_layer = next_layer

        subgraph = self.graph.subgraph(sub_nodes)
        
        triplets = []
        for u, v, data in subgraph.edges(data=True):
            triplets.append(GraphTriplet(
                subject=str(u),
                relation=data.get("relation", "connected_to"),
                object=str(v),
                confidence=data.get("confidence", 1.0),
                source_doc_id=data.get("source_doc_id", "")
            ))
            
        return SubGraphContext(
            target_entity=target_entity,
            triplets=triplets,
            neighbor_nodes=[str(n) for n in sub_nodes]
        )

    def query_graph_context(self, query_text: str, hops: int = 2) -> Tuple[List[GraphTriplet], List[str]]:
        """Multi-entity subgraph extraction for user query."""
        matched_entities = self.find_matching_entities(query_text)
        all_triplets: List[GraphTriplet] = []
        all_entities: Set[str] = set(matched_entities)
        
        seen_triplet_keys = set()
        for entity in matched_entities:
            sub = self.get_subgraph(entity, hops=hops)
            all_entities.update(sub.neighbor_nodes)
            for t in sub.triplets:
                key = (t.subject, t.relation, t.object)
                if key not in seen_triplet_keys:
                    seen_triplet_keys.add(key)
                    all_triplets.append(t)

        return all_triplets, list(all_entities)

    def generate_pyvis_html(self, triplets: List[GraphTriplet], highlight_nodes: Optional[List[str]] = None) -> str:
        """Generate interactive HTML for PyVis visualizer in Streamlit."""
        try:
            from pyvis.network import Network
            net = Network(height="450px", width="100%", bgcolor="#0F172A", font_color="#F8FAFC", directed=True)
            
            highlight_set = set(highlight_nodes or [])
            added_nodes = set()
            
            for t in triplets:
                for n in [t.subject, t.object]:
                    if n not in added_nodes:
                        color = "#3B82F6" if n in highlight_set else "#64748B"
                        size = 22 if n in highlight_set else 16
                        net.add_node(n, label=n, color=color, size=size)
                        added_nodes.add(n)
                
                net.add_edge(t.subject, t.object, title=t.relation, label=t.relation, color="#94A3B8")
                
            net.toggle_physics(True)
            return net.generate_html()
        except Exception as e:
            logger.error(f"Error generating PyVis graph HTML: {e}")
            return self._fallback_html_graph(triplets)

    def _fallback_html_graph(self, triplets: List[GraphTriplet]) -> str:
        """Simple HTML SVG graph fallback if PyVis has issues."""
        nodes = set()
        for t in triplets:
            nodes.add(t.subject)
            nodes.add(t.object)
        
        items_html = "".join([f"<li style='color:#38BDF8;'><b>{t.subject}</b> &rarr; <i>[{t.relation}]</i> &rarr; <b>{t.object}</b></li>" for t in triplets])
        return f"""
        <div style="background-color:#1E293B; padding:15px; border-radius:8px; color:#F8FAFC;">
            <h4 style="margin-top:0; color:#3B82F6;">Knowledge Sub-Graph Provenance ({len(triplets)} relations)</h4>
            <ul style="max-height:300px; overflow-y:auto; line-height:1.6;">
                {items_html or "<li>No graph relations retrieved for query.</li>"}
            </ul>
        </div>
        """

    def export_graphml(self) -> str:
        """Export Knowledge Graph to standard GraphML XML string representation."""
        import io
        buffer = io.BytesIO()
        nx.write_graphml(self.graph, buffer)
        return buffer.getvalue().decode("utf-8")

    def export_cypher(self) -> str:
        """Export Knowledge Graph to Neo4j Cypher import script."""
        cypher_lines = ["// Neo4j Cypher Knowledge Graph Import Script", "// Generated by GraphRAG Enterprise Engine", ""]
        
        # Create Nodes
        for node in self.graph.nodes():
            safe_node = str(node).replace("'", "\\'")
            node_data = self.graph.nodes[node]
            node_type = node_data.get("type", "Entity")
            cypher_lines.append(f"MERGE (n:`{node_type}` {{id: '{safe_node}', name: '{safe_node}'}})")
            
        cypher_lines.append("")
        # Create Relationships
        for u, v, data in self.graph.edges(data=True):
            safe_u = str(u).replace("'", "\\'")
            safe_v = str(v).replace("'", "\\'")
            rel = data.get("relation", "REL")
            # Replace invalid Cypher rel chars
            safe_rel = rel.upper().replace(" ", "_").replace("-", "_")
            cypher_lines.append(
                f"MATCH (a {{id: '{safe_u}'}}), (b {{id: '{safe_v}'}}) "
                f"MERGE (a)-[r:`{safe_rel}`]->(b)"
            )
            
        return "\n".join(cypher_lines)

    def get_stats(self) -> Dict[str, int]:
        """Return total count of nodes and edges in graph."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges()
        }

    def save_graph(self):
        """Serialize NetworkX graph to JSON file."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "nodes": [{"id": n, **self.graph.nodes[n]} for n in self.graph.nodes()],
            "edges": [{"subject": u, "object": v, **d} for u, v, d in self.graph.edges(data=True)]
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_graph(self):
        """Load graph from JSON file if it exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for node in data.get("nodes", []):
                    node_id = node.pop("id")
                    self.graph.add_node(node_id, **node)
                for edge in data.get("edges", []):
                    subj = edge.pop("subject")
                    obj = edge.pop("object")
                    self.graph.add_edge(subj, obj, **edge)
                logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            except Exception as e:
                logger.error(f"Error loading graph from {self.storage_path}: {e}")
