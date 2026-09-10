import logging
import json
import requests
from typing import List, Dict, Any, Optional
from src.models import CRAGResult, VectorChunk, GraphTriplet
from src.llm_client import GeminiClient
from src.config import settings

logger = logging.getLogger(__name__)

# Try importing tavily
try:
    from tavily import TavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False
    logger.warning("tavily-python not installed. Web search will use fallback HTTP / mock search.")

class CRAGEngine:
    def __init__(self, llm_client: GeminiClient, confidence_threshold: Optional[float] = None):
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold or settings.crag_confidence_threshold
        self.tavily_client = None
        
        if HAS_TAVILY and settings.tavily_api_key:
            try:
                self.tavily_client = TavilyClient(api_key=settings.tavily_api_key)
                logger.info("Initialized Tavily Web Search Client.")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily Client: {e}")

    def evaluate_context(self, query: str, context_str: str) -> CRAGResult:
        """Evaluate retrieved context relevance and calculate confidence score."""
        if not context_str.strip():
            return CRAGResult(
                confidence_score=0.1,
                grade="Incorrect",
                web_fallback_triggered=True,
                web_query=query
            )

        prompt = f"""Evaluate the relevance of the following RETRIEVED CONTEXT for answering the USER QUERY.
Score the context relevance on a scale from 0.0 to 1.0.
Output ONLY a JSON object with keys:
- "confidence_score": float between 0.0 and 1.0
- "grade": string ("Correct" if score >= 0.65, "Ambiguous" if 0.4 <= score < 0.65, "Incorrect" if score < 0.4)
- "reasoning": brief explanation

USER QUERY: "{query}"
RETRIEVED CONTEXT:
{context_str[:2000]}
"""
        try:
            res_text = self.llm_client.generate_text(prompt)
            clean_json = self.llm_client._clean_json_string(res_text)
            data = json.loads(clean_json)
            score = float(data.get("confidence_score", 0.75))
            grade = data.get("grade", "Correct")
            
            trigger_fallback = score < self.confidence_threshold
            
            return CRAGResult(
                confidence_score=score,
                grade=grade,
                web_fallback_triggered=trigger_fallback,
                web_query=query if trigger_fallback else None
            )
        except Exception as e:
            logger.warning(f"Error in CRAG LLM grading: {e}. Performing heuristic score calculation.")
            return self._heuristic_evaluate(query, context_str)

    def search_web_fallback(self, query: str) -> List[str]:
        """Perform external web search fallback using Tavily API or mock search."""
        snippets = []
        if self.tavily_client:
            try:
                res = self.tavily_client.search(query=query, max_results=3)
                for item in res.get("results", []):
                    snippets.append(f"[Web Result: {item.get('title', 'Web')}]\n{item.get('content', '')}")
                if snippets:
                    return snippets
            except Exception as e:
                logger.error(f"Tavily web search error: {e}")

        # Fallback web search simulation
        return [
            f"[Simulated Web Search Fallback for: '{query}']\nGlobal industry standards, USMCA compliance guidelines, and semiconductor dual-sourcing framework updates confirm key supply chain resilience protocols."
        ]

    def _heuristic_evaluate(self, query: str, context_str: str) -> CRAGResult:
        """Heuristic relevance scorer when LLM evaluation is unavailable."""
        import re
        # Normalize context text (replace underscores with spaces for relation matching)
        normalized_context = context_str.replace('_', ' ')
        
        query_words = [w.lower() for w in re.findall(r'\b[\w-]+\b', query) if len(w) > 2]
        context_words = set(w.lower() for w in re.findall(r'\b[\w-]+\b', normalized_context))
        
        if not query_words:
            score = 0.75
        else:
            matches = 0
            for qw in query_words:
                # Check exact or prefix match (stemming heuristic)
                if qw in context_words or any(cw.startswith(qw[:4]) for cw in context_words if len(cw) >= 4 and len(qw) >= 4):
                    matches += 1
            overlap_ratio = matches / len(query_words)
            score = min(1.0, max(0.2, overlap_ratio * 1.4))

        score = round(score, 2)
        if score >= 0.65:
            grade = "Correct"
        elif score >= 0.4:
            grade = "Ambiguous"
        else:
            grade = "Incorrect"

        trigger = score < self.confidence_threshold

        return CRAGResult(
            confidence_score=score,
            grade=grade,
            web_fallback_triggered=trigger,
            web_query=query if trigger else None
        )
