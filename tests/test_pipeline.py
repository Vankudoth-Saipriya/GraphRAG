import sys
import os
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models import Document, GraphTriplet, QueryRequest
from src.llm_client import GeminiClient
from src.graph_engine import GraphEngine
from src.vector_engine import VectorEngine
from src.hybrid_retriever import HybridRetriever
from src.crag_engine import CRAGEngine
from src.eval_engine import BenchmarkEvaluator

class TestGraphRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.llm_client = GeminiClient()
        cls.graph_engine = GraphEngine(storage_path=str(BASE_DIR / "data" / "test_graph.json"))
        cls.vector_engine = VectorEngine(llm_client=cls.llm_client)
        cls.hybrid_retriever = HybridRetriever(
            graph_engine=cls.graph_engine,
            vector_engine=cls.vector_engine,
            llm_client=cls.llm_client
        )
        cls.crag_engine = CRAGEngine(llm_client=cls.llm_client)

    def test_01_graph_engine_ingest_and_query(self):
        triplets = [
            GraphTriplet(subject="ADU-7", relation="uses", object="NX-9000"),
            GraphTriplet(subject="NX-9000", relation="supplied_by", object="NexusChip Corp")
        ]
        self.graph_engine.add_triplets(triplets)
        
        sub = self.graph_engine.get_subgraph("ADU-7", hops=2)
        self.assertGreaterEqual(len(sub.triplets), 1)
        self.assertIn("NX-9000", sub.neighbor_nodes)

    def test_02_vector_engine_ingest_and_search(self):
        doc = Document(
            id="TEST-DOC-1",
            title="Test Microcontroller Spec",
            category="Test",
            content="NX-9000 microcontroller is manufactured by NexusChip Corp for ADU-7 unit."
        )
        chunks_count = self.vector_engine.ingest_documents([doc])
        self.assertGreaterEqual(chunks_count, 1)

        results = self.vector_engine.search("Who manufactures NX-9000?", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("NexusChip Corp", results[0].text)

    def test_03_hybrid_retriever(self):
        v_chunks, g_triplets, entities = self.hybrid_retriever.retrieve(
            query_text="Who supplies NX-9000 for ADU-7?",
            mode="hybrid",
            top_k=2,
            hops=2
        )
        self.assertIsNotNone(v_chunks)
        self.assertIsNotNone(g_triplets)
        
        context_str = self.hybrid_retriever.format_context(v_chunks, g_triplets)
        self.assertTrue(len(context_str) > 0)
        
        answer = self.hybrid_retriever.generate_answer("Who supplies NX-9000 for ADU-7?", context_str)
        self.assertIsNotNone(answer)
        self.assertTrue(len(answer) > 10)

    def test_04_crag_engine_evaluation(self):
        context = "### KNOWLEDGE GRAPH FACT TRIPLETS:\n- (ADU-7) --[uses]--> (NX-9000)\n- (NX-9000) --[supplied_by]--> (NexusChip Corp)"
        res = self.crag_engine.evaluate_context("Who supplies microcontrollers for ADU-7?", context)
        self.assertGreaterEqual(res.confidence_score, 0.4)
        self.assertIn(res.grade, ["Correct", "Ambiguous"])

    def test_05_benchmark_evaluator(self):
        evaluator = BenchmarkEvaluator(self.hybrid_retriever, self.crag_engine)
        metrics = evaluator.evaluate_mode(mode="hybrid", max_questions=2)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_questions, 2)
        self.assertGreaterEqual(metrics.context_recall, 0.0)

    def test_06_fuzzy_entity_matching(self):
        # Test fuzzy matching with RapidFuzz for slight entity misspellings
        matched = self.graph_engine.find_matching_entities("Who makes NexusChip microcontrollers?", score_threshold=60.0)
        self.assertGreaterEqual(len(matched), 1)

    def test_07_graph_export(self):
        graphml_xml = self.graph_engine.export_graphml()
        self.assertIn("<graphml", graphml_xml)
        
        cypher_script = self.graph_engine.export_cypher()
        self.assertIn("MERGE", cypher_script)

if __name__ == "__main__":
    unittest.main()

