import logging
import time
from typing import List, Dict, Any, Tuple
from src.models import (
    QueryRequest, QueryResponse, Provenance, VectorChunk, GraphTriplet
)
from src.graph_engine import GraphEngine
from src.vector_engine import VectorEngine
from src.llm_client import GeminiClient

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, graph_engine: GraphEngine, vector_engine: VectorEngine, llm_client: GeminiClient):
        self.graph_engine = graph_engine
        self.vector_engine = vector_engine
        self.llm_client = llm_client

    def retrieve(self, query_text: str, mode: str = "hybrid", top_k: int = 4, hops: int = 2) -> Tuple[List[VectorChunk], List[GraphTriplet], List[str]]:
        """Retrieve vector chunks and graph sub-graph triplets based on mode."""
        vector_chunks: List[VectorChunk] = []
        graph_triplets: List[GraphTriplet] = []
        matched_entities: List[str] = []

        if mode in ["hybrid", "vector"]:
            vector_chunks = self.vector_engine.search(query_text, top_k=top_k)

        if mode in ["hybrid", "graph"]:
            graph_triplets, matched_entities = self.graph_engine.query_graph_context(query_text, hops=hops)

        return vector_chunks, graph_triplets, matched_entities

    def format_context(self, vector_chunks: List[VectorChunk], graph_triplets: List[GraphTriplet]) -> str:
        """Format retrieved vector text chunks and graph triplets into unified prompt context."""
        context_parts = []

        if graph_triplets:
            triplet_str = "\n".join([f"- ({t.subject}) --[{t.relation}]--> ({t.object})" for t in graph_triplets])
            context_parts.append(f"### KNOWLEDGE GRAPH FACT TRIPLETS:\n{triplet_str}")

        if vector_chunks:
            chunks_str = "\n\n".join([f"[Source: {c.doc_title}]\n{c.text}" for c in vector_chunks])
            context_parts.append(f"### DOCUMENT TEXT CONTEXT:\n{chunks_str}")

        return "\n\n".join(context_parts)

    def generate_answer(self, query: str, context_str: str) -> str:
        """Generate final answer using Gemini LLM given unified context."""
        prompt = f"""Synthesize an accurate, professional answer for the user query based ONLY on the provided context.
Cite specific entities, relationships, suppliers, compliance requirements, or components whenever mentioned.

USER QUERY:
"{query}"

RETRIEVED GRAPH & DOCUMENT CONTEXT:
{context_str}

ANSWER:
"""
        system_prompt = "You are an Enterprise GraphRAG Assistant. You provide precise, ground-truth answers backed by knowledge graph relationships and vector documentation."
        return self.llm_client.generate_text(prompt, system_instruction=system_prompt)
