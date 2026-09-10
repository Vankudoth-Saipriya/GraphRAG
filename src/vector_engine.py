import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from src.models import Document, VectorChunk
from src.llm_client import GeminiClient
from src.config import settings

logger = logging.getLogger(__name__)

# Try importing ChromaDB
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except Exception as e:
    HAS_CHROMADB = False
    logger.warning(f"ChromaDB not available: {e}. Falling back to in-memory cosine vector store.")

class VectorEngine:
    def __init__(self, llm_client: GeminiClient, persist_dir: Optional[str] = None):
        self.llm_client = llm_client
        self.persist_dir = persist_dir or settings.vector_db_dir
        self.chroma_client = None
        self.collection = None
        
        # Memory fallback store
        self.chunks_memory: List[VectorChunk] = []
        self.embeddings_memory: List[List[float]] = []

        if HAS_CHROMADB:
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = self.chroma_client.get_or_create_collection(
                    name="enterprise_docs",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Successfully initialized ChromaDB collection.")
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}. Switching to in-memory vector store.")
                self.chroma_client = None

    def chunk_document(self, doc: Document, chunk_size: int = 400, overlap: int = 50) -> List[VectorChunk]:
        """Split document content into overlapping text chunks."""
        words = doc.content.split()
        chunks = []
        
        if len(words) <= chunk_size:
            chunks.append(VectorChunk(
                chunk_id=f"{doc.id}_chunk_0",
                doc_id=doc.id,
                doc_title=doc.title,
                text=doc.content,
                similarity_score=0.0
            ))
            return chunks

        start = 0
        chunk_idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(VectorChunk(
                chunk_id=f"{doc.id}_chunk_{chunk_idx}",
                doc_id=doc.id,
                doc_title=doc.title,
                text=chunk_text,
                similarity_score=0.0
            ))
            chunk_idx += 1
            start += chunk_size - overlap
            
        return chunks

    def ingest_documents(self, docs: List[Document]) -> int:
        """Chunk documents, embed, and index into ChromaDB / memory store."""
        all_chunks: List[VectorChunk] = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        texts = [c.text for c in all_chunks]
        embeddings = self.llm_client.get_embeddings_batch(texts)

        if self.collection:
            try:
                ids = [c.chunk_id for c in all_chunks]
                metadatas = [{"doc_id": c.doc_id, "doc_title": c.doc_title} for c in all_chunks]
                
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                logger.info(f"Indexed {len(all_chunks)} chunks into ChromaDB.")
            except Exception as e:
                logger.error(f"ChromaDB indexing error: {e}. Indexing to in-memory store.")
                self._ingest_memory(all_chunks, embeddings)
        else:
            self._ingest_memory(all_chunks, embeddings)

        return len(all_chunks)

    def search(self, query: str, top_k: int = 4) -> List[VectorChunk]:
        """Perform vector similarity search for top_k document chunks."""
        query_embedding = self.llm_client.get_embedding(query)

        if self.collection and self.collection.count() > 0:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.collection.count())
                )
                
                vector_chunks = []
                if results and results.get("documents") and len(results["documents"]) > 0:
                    docs = results["documents"][0]
                    ids = results["ids"][0]
                    metadatas = results["metadatas"][0]
                    distances = results.get("distances", [[0.0] * len(docs)])[0]
                    
                    for i in range(len(docs)):
                        # Cosine similarity score = 1 - cosine distance
                        sim_score = max(0.0, 1.0 - distances[i]) if distances else 0.8
                        vector_chunks.append(VectorChunk(
                            chunk_id=ids[i],
                            doc_id=metadatas[i].get("doc_id", "unknown"),
                            doc_title=metadatas[i].get("doc_title", "Untitled"),
                            text=docs[i],
                            similarity_score=float(sim_score)
                        ))
                return vector_chunks
            except Exception as e:
                logger.error(f"ChromaDB search failed: {e}. Falling back to memory store search.")
                return self._search_memory(query_embedding, top_k)
        
        return self._search_memory(query_embedding, top_k)

    def get_chunk_count(self) -> int:
        """Return total chunk count."""
        if self.collection:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(self.chunks_memory)

    def _ingest_memory(self, chunks: List[VectorChunk], embeddings: List[List[float]]):
        """Index chunks into memory store."""
        for c, emb in zip(chunks, embeddings):
            self.chunks_memory.append(c)
            self.embeddings_memory.append(emb)

    def _search_memory(self, query_emb: List[float], top_k: int) -> List[VectorChunk]:
        """In-memory cosine similarity search."""
        if not self.embeddings_memory:
            return []

        q_vec = np.array(query_emb)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        scores = []
        for idx, emb in enumerate(self.embeddings_memory):
            e_vec = np.array(emb)
            e_norm = np.linalg.norm(e_vec)
            if e_norm == 0:
                e_norm = 1.0
            sim = np.dot(q_vec, e_vec) / (q_norm * e_norm)
            scores.append((sim, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_results = scores[:top_k]

        results = []
        for score, idx in top_results:
            chunk = self.chunks_memory[idx]
            results.append(VectorChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                text=chunk.text,
                similarity_score=float(score)
            ))
        return results
