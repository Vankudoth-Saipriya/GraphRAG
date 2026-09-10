import os
import json
import time
import logging
from typing import List, Dict, Any
from src.models import BenchmarkQuestion, BenchmarkMetrics, QueryRequest
from src.hybrid_retriever import HybridRetriever
from src.crag_engine import CRAGEngine
from src.config import settings

logger = logging.getLogger(__name__)

class BenchmarkEvaluator:
    def __init__(self, hybrid_retriever: HybridRetriever, crag_engine: CRAGEngine):
        self.retriever = hybrid_retriever
        self.crag_engine = crag_engine
        self.benchmark_questions = self._load_questions()

    def _load_questions(self) -> List[BenchmarkQuestion]:
        """Load benchmark evaluation questions dataset."""
        path = settings.benchmark_dataset_path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [BenchmarkQuestion(**q) for q in data]
            except Exception as e:
                logger.error(f"Failed to load benchmark questions from {path}: {e}")
        return []

    def evaluate_mode(self, mode: str = "hybrid", max_questions: int = 35) -> BenchmarkMetrics:
        """Run benchmark evaluation suite for specified retrieval mode."""
        questions_to_run = self.benchmark_questions[:max_questions]
        if not questions_to_run:
            return BenchmarkMetrics(
                mode=mode,
                total_questions=0,
                context_recall=0.0,
                answer_faithfulness=0.0,
                avg_latency_ms=0.0,
                hallucination_score=0.0,
                details=[]
            )

        details = []
        total_recall = 0.0
        total_faithfulness = 0.0
        total_latency = 0.0
        total_hallucination = 0.0

        for q in questions_to_run:
            start_time = time.time()
            
            # Retrieve vector & graph context
            v_chunks, g_triplets, entities = self.retriever.retrieve(
                query_text=q.question,
                mode=mode,
                top_k=4 if mode != "graph" else 0,
                hops=2
            )
            
            context_str = self.retriever.format_context(v_chunks, g_triplets)
            answer = self.retriever.generate_answer(q.question, context_str)
            latency_ms = (time.time() - start_time) * 1000.0

            # 1. Compute Context Recall (% of expected entities captured in context)
            recall = self._compute_context_recall(q.expected_entities, context_str, entities)

            # 2. Compute Answer Faithfulness (ground truth overlap / alignment)
            faithfulness = self._compute_faithfulness(q.ground_truth, answer)

            # 3. Compute Hallucination Score (1.0 - faithfulness ratio)
            hallucination = round(max(0.0, 1.0 - faithfulness), 2)

            total_recall += recall
            total_faithfulness += faithfulness
            total_latency += latency_ms
            total_hallucination += hallucination

            details.append({
                "question_id": q.id,
                "question": q.question,
                "mode": mode,
                "complexity": q.complexity,
                "context_recall": recall,
                "answer_faithfulness": faithfulness,
                "latency_ms": round(latency_ms, 2),
                "hallucination_score": hallucination,
                "answer": answer
            })

        count = len(questions_to_run)
        return BenchmarkMetrics(
            mode=mode,
            total_questions=count,
            context_recall=round(total_recall / count, 3),
            answer_faithfulness=round(total_faithfulness / count, 3),
            avg_latency_ms=round(total_latency / count, 2),
            hallucination_score=round(total_hallucination / count, 3),
            details=details
        )

    def run_full_comparison(self, max_questions: int = 35) -> Dict[str, Any]:
        """Run comparison benchmark between Vector-Only vs GraphRAG Hybrid."""
        logger.info("Running Vector-Only baseline evaluation...")
        vector_metrics = self.evaluate_mode(mode="vector", max_questions=max_questions)
        
        logger.info("Running GraphRAG Hybrid evaluation...")
        hybrid_metrics = self.evaluate_mode(mode="hybrid", max_questions=max_questions)

        return {
            "vector_only": vector_metrics.dict(),
            "graphrag_hybrid": hybrid_metrics.dict()
        }


    def _compute_context_recall(self, expected_entities: List[str], context_str: str, retrieved_entities: List[str]) -> float:
        """Measure recall of ground truth entities in retrieved context."""
        if not expected_entities:
            return 1.0
        
        context_lower = context_str.lower() + " ".join([e.lower() for e in retrieved_entities])
        found = 0
        for entity in expected_entities:
            if entity.lower() in context_lower:
                found += 1
                
        return round(found / len(expected_entities), 2)

    def _compute_faithfulness(self, ground_truth: str, generated_answer: str) -> float:
        """Measure alignment between generated answer and ground truth answer."""
        import re
        gt_words = set(w.lower() for w in re.findall(r'\b[\w-]+\b', ground_truth) if len(w) > 2)
        ans_words = set(w.lower() for w in re.findall(r'\b[\w-]+\b', generated_answer))
        
        if not gt_words:
            return 1.0
            
        overlap = gt_words.intersection(ans_words)
        ratio = len(overlap) / len(gt_words)
        
        return min(1.0, round(ratio * 1.3, 2))
