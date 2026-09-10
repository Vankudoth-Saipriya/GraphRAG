from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    category: str = Field("General", description="Category or domain")
    content: str = Field(..., description="Full text content of document")

class EntityNode(BaseModel):
    id: str = Field(..., description="Unique entity name/ID")
    type: str = Field("Entity", description="Entity type, e.g. Product, Supplier, Regulation, Region")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Metadata key-values")

class GraphTriplet(BaseModel):
    subject: str = Field(..., description="Source entity name")
    relation: str = Field(..., description="Relationship label")
    object: str = Field(..., description="Target entity name")
    confidence: float = Field(1.0, description="Extraction confidence score")
    source_doc_id: Optional[str] = Field(None, description="Source document ID")

class KnowledgeGraphExport(BaseModel):
    nodes: List[EntityNode] = Field(default_factory=list)
    edges: List[GraphTriplet] = Field(default_factory=list)

class VectorChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    similarity_score: float

class SubGraphContext(BaseModel):
    target_entity: str
    triplets: List[GraphTriplet]
    neighbor_nodes: List[str]

class CRAGResult(BaseModel):
    confidence_score: float
    grade: str = Field("Correct", description="Correct, Ambiguous, or Incorrect")
    web_fallback_triggered: bool = False
    web_query: Optional[str] = None
    web_snippets: List[str] = Field(default_factory=list)

class Provenance(BaseModel):
    vector_chunks: List[VectorChunk] = Field(default_factory=list)
    graph_triplets: List[GraphTriplet] = Field(default_factory=list)
    retrieved_entities: List[str] = Field(default_factory=list)
    crag_evaluation: Optional[CRAGResult] = None

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    mode: str = Field("hybrid", description="Retrieval mode: hybrid, graph, or vector")
    top_k_vector: int = Field(4, description="Top K vector chunks")
    hops: int = Field(2, description="N-hop graph traversal depth")
    enable_crag: bool = Field(True, description="Enable Corrective RAG evaluation")

class QueryResponse(BaseModel):
    query: str
    answer: str
    mode: str
    provenance: Provenance
    execution_time_ms: float

class IngestResponse(BaseModel):
    status: str
    docs_processed: int
    chunks_created: int
    nodes_added: int
    edges_added: int

class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    ground_truth: str
    expected_entities: List[str] = Field(default_factory=list)
    complexity: str = "Multi-Hop"

class BenchmarkMetrics(BaseModel):
    mode: str
    total_questions: int
    context_recall: float
    answer_faithfulness: float
    avg_latency_ms: float
    hallucination_score: float
    details: List[Dict[str, Any]] = Field(default_factory=list)
