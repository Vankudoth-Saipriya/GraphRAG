import time
import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, PlainTextResponse

from src.models import (
    Document, QueryRequest, QueryResponse, IngestResponse,
    Provenance, CRAGResult, BenchmarkMetrics, GraphTriplet
)
from src.config import settings
from src.llm_client import GeminiClient
from src.graph_engine import GraphEngine
from src.vector_engine import VectorEngine
from src.hybrid_retriever import HybridRetriever
from src.crag_engine import CRAGEngine
from src.eval_engine import BenchmarkEvaluator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag_api")

app = FastAPI(
    title="GraphRAG Enterprise Engine API",
    description="Knowledge Graph + ChromaDB Vector Retrieval Engine with Corrective RAG (CRAG) & Benchmark Suite",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine Services Initialization
llm_client = GeminiClient()
graph_engine = GraphEngine()
vector_engine = VectorEngine(llm_client=llm_client)
hybrid_retriever = HybridRetriever(graph_engine=graph_engine, vector_engine=vector_engine, llm_client=llm_client)
crag_engine = CRAGEngine(llm_client=llm_client)
evaluator = BenchmarkEvaluator(hybrid_retriever=hybrid_retriever, crag_engine=crag_engine)

@app.on_event("startup")
def auto_ingest_on_startup():
    """Auto-ingest synthetic dataset on API startup if graph is empty."""
    if graph_engine.graph.number_of_nodes() == 0:
        logger.info("Initializing auto-ingestion of synthetic dataset...")
        if os.path.exists(settings.docs_dataset_path):
            with open(settings.docs_dataset_path, "r", encoding="utf-8") as f:
                docs_data = json.load(f)
            docs = [Document(**d) for d in docs_data]
            
            # Ingest to vector engine
            vector_engine.ingest_documents(docs)
            
            # Ingest to graph engine
            for doc in docs:
                triplets = llm_client.extract_triplets(doc.content, doc.id)
                graph_engine.add_triplets(triplets)
                
            logger.info(f"Auto-ingested {len(docs)} synthetic documents successfully.")

@app.get("/")
def root_index():
    """Root endpoint welcoming users and directing to interactive API docs."""
    return {
        "title": "GraphRAG Enterprise Engine API",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }

@app.get("/api/v1/health")
def health_check():
    """Check API status and database statistics."""
    stats = graph_engine.get_stats()
    return {
        "status": "healthy",
        "environment": settings.env,
        "nodes_count": stats["num_nodes"],
        "edges_count": stats["num_edges"],
        "vector_chunks_count": vector_engine.get_chunk_count(),
        "has_gemini_key": bool(settings.gemini_api_key),
        "has_tavily_key": bool(settings.tavily_api_key)
    }

@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_documents(docs: List[Document]):
    """Ingest documents into Knowledge Graph (NetworkX) and Vector Database (ChromaDB)."""
    try:
        chunks_count = vector_engine.ingest_documents(docs)
        
        total_triplets = []
        for doc in docs:
            triplets = llm_client.extract_triplets(doc.content, doc.id)
            total_triplets.extend(triplets)
            
        initial_nodes = graph_engine.graph.number_of_nodes()
        initial_edges = graph_engine.graph.number_of_edges()
        
        graph_engine.add_triplets(total_triplets)
        
        nodes_added = graph_engine.graph.number_of_nodes() - initial_nodes
        edges_added = graph_engine.graph.number_of_edges() - initial_edges
        
        return IngestResponse(
            status="success",
            docs_processed=len(docs),
            chunks_created=chunks_count,
            nodes_added=max(0, nodes_added),
            edges_added=max(0, edges_added)
        )
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query", response_model=QueryResponse)
def query_graphrag(req: QueryRequest):
    """Execute GraphRAG query with hybrid retrieval, CRAG scoring, and provenance tracking."""
    start_time = time.time()
    try:
        # Retrieve vector chunks & graph triplets
        v_chunks, g_triplets, matched_entities = hybrid_retriever.retrieve(
            query_text=req.query,
            mode=req.mode,
            top_k=req.top_k_vector,
            hops=req.hops
        )

        context_str = hybrid_retriever.format_context(v_chunks, g_triplets)
        crag_result = None

        if req.enable_crag:
            crag_result = crag_engine.evaluate_context(req.query, context_str)
            if crag_result.web_fallback_triggered:
                web_snippets = crag_engine.search_web_fallback(req.query)
                crag_result.web_snippets = web_snippets
                web_context_str = "\n".join(web_snippets)
                context_str += f"\n\n### EXTERNAL WEB SEARCH CONTEXT (CRAG FALLBACK):\n{web_context_str}"

        answer = hybrid_retriever.generate_answer(req.query, context_str)
        exec_time_ms = (time.time() - start_time) * 1000.0

        provenance = Provenance(
            vector_chunks=v_chunks,
            graph_triplets=g_triplets,
            retrieved_entities=matched_entities,
            crag_evaluation=crag_result
        )

        return QueryResponse(
            query=req.query,
            answer=answer,
            mode=req.mode,
            provenance=provenance,
            execution_time_ms=round(exec_time_ms, 2)
        )
    except Exception as e:
        logger.error(f"Error executing GraphRAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/eval")
def run_evaluation(max_questions: int = Query(35, ge=1, le=35)):
    """Run benchmark evaluation suite comparing Vector-Only RAG vs GraphRAG Hybrid."""
    try:
        results = evaluator.run_full_comparison(max_questions=max_questions)
        return {
            "status": "success",
            "benchmark_results": results
        }
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/graph/stats")
def get_graph_stats():
    """Return total node and edge count of the Knowledge Graph."""
    return graph_engine.get_stats()

@app.get("/api/v1/graph/pyvis", response_class=HTMLResponse)
def get_graph_pyvis_html(query: Optional[str] = None):
    """Return PyVis network visualization HTML string for requested query neighborhood."""
    if query:
        triplets, matched = graph_engine.query_graph_context(query, hops=2)
        html = graph_engine.generate_pyvis_html(triplets, highlight_nodes=matched)
    else:
        # Whole graph preview up to top 50 edges
        all_triplets = []
        for u, v, d in list(graph_engine.graph.edges(data=True))[:50]:
            all_triplets.append(GraphTriplet(subject=str(u), relation=d.get("relation", "connected"), object=str(v)))
        html = graph_engine.generate_pyvis_html(all_triplets)
    return HTMLResponse(content=html)

@app.get("/api/v1/graph/export/graphml", response_class=PlainTextResponse)
def export_graphml():
    """Export Knowledge Graph to standard GraphML XML string."""
    try:
        graphml_xml = graph_engine.export_graphml()
        return Response(content=graphml_xml, media_type="application/xml", headers={"Content-Disposition": "attachment; filename=knowledge_graph.graphml"})
    except Exception as e:
        logger.error(f"Error exporting GraphML: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/graph/export/cypher", response_class=PlainTextResponse)
def export_cypher():
    """Export Knowledge Graph to Neo4j Cypher import script."""
    try:
        cypher_script = graph_engine.export_cypher()
        return Response(content=cypher_script, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=knowledge_graph.cypher"})
    except Exception as e:
        logger.error(f"Error exporting Cypher script: {e}")
        raise HTTPException(status_code=500, detail=str(e))
