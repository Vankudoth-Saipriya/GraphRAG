import os
import json
import time
import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="GraphRAG Enterprise Engine",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme glassmorphism & visual excellence
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 50%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Cards and Glassmorphism */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .provenance-card {
        background: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-radius: 4px;
    }
    
    .crag-badge-correct {
        background-color: #10B981;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .crag-badge-ambiguous {
        background-color: #F59E0B;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .crag-badge-incorrect {
        background-color: #EF4444;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Direct Engine Import for Single-Process Cloud Deployment Fallback
@st.cache_resource
def get_local_engines():
    try:
        from src.config import settings
        from src.llm_client import GeminiClient
        from src.graph_engine import GraphEngine
        from src.vector_engine import VectorEngine
        from src.hybrid_retriever import HybridRetriever
        from src.crag_engine import CRAGEngine
        from src.eval_engine import BenchmarkEvaluator
        from src.models import Document, QueryRequest
        
        llm = GeminiClient()
        graph = GraphEngine()
        vector = VectorEngine(llm_client=llm)
        
        # Auto-ingest if empty
        if graph.graph.number_of_nodes() == 0 and os.path.exists(settings.docs_dataset_path):
            with open(settings.docs_dataset_path, "r", encoding="utf-8") as f:
                docs = [Document(**d) for d in json.load(f)]
            vector.ingest_documents(docs)
            for d in docs:
                triplets = llm.extract_triplets(d.content, d.id)
                graph.add_triplets(triplets)

        retriever = HybridRetriever(graph_engine=graph, vector_engine=vector, llm_client=llm)
        crag = CRAGEngine(llm_client=llm)
        evaluator = BenchmarkEvaluator(hybrid_retriever=retriever, crag_engine=crag)

        return {
            "graph": graph,
            "vector": vector,
            "retriever": retriever,
            "crag": crag,
            "evaluator": evaluator,
            "llm": llm
        }
    except Exception as e:
        st.error(f"Engine initialization error: {e}")
        return None

engines = get_local_engines()

# Helper for API vs Direct Python execution
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

def check_backend_api():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None

use_api, backend_health = check_backend_api()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/network.png", width=64)
    st.title("GraphRAG Engine")
    st.markdown("**Enterprise Knowledge Graph + Corrective RAG (CRAG)**")
    st.divider()

    # System Stats
    if use_api and backend_health:
        st.success("🟢 API Server Connected")
        nodes_cnt = backend_health.get("nodes_count", 0)
        edges_cnt = backend_health.get("edges_count", 0)
        chunks_cnt = backend_health.get("vector_chunks_count", 0)
    elif engines:
        st.info("⚡ Single-Process Mode (Cloud)")
        nodes_cnt = engines["graph"].graph.number_of_nodes()
        edges_cnt = engines["graph"].graph.number_of_edges()
        chunks_cnt = engines["vector"].get_chunk_count()
    else:
        nodes_cnt, edges_cnt, chunks_cnt = 0, 0, 0

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Graph Nodes", nodes_cnt)
    with col_s2:
        st.metric("Graph Edges", edges_cnt)
    st.metric("Vector Chunks", chunks_cnt)

    st.divider()
    st.markdown("### Settings")
    retrieval_mode = st.selectbox(
        "Retrieval Engine Mode",
        options=["hybrid", "vector", "graph"],
        format_func=lambda x: "🕸️ Hybrid (Graph + Vector)" if x == "hybrid" else ("🔍 Vector-Only Baseline" if x == "vector" else "🌐 Knowledge Graph Only"),
        index=0
    )
    enable_crag = st.checkbox("Enable Corrective RAG (CRAG)", value=True)
    top_k = st.slider("Top-K Vector Chunks", min_value=1, max_value=8, value=4)
    hops = st.slider("Graph Traversal Hops", min_value=1, max_value=3, value=2)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "💬 Q&A Studio & Sub-Graph Visualizer",
    "📂 Knowledge Base Ingestion",
    "📊 RAGAS Benchmark & Analytics"
])

# -----------------------------------------------------------------------------
# TAB 1: Q&A STUDIO
# -----------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="main-title">GraphRAG Interactive Q&A Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-hop enterprise query synthesis with sub-graph provenance and Corrective RAG fallback</div>', unsafe_allow_html=True)

    # Sample Questions Dropdown for Quick Demonstration
    sample_questions = [
        "Select a sample question...",
        "Who is the primary microcontroller supplier for ADU-7, and what is the secondary fallback contingency?",
        "If LibCrypto-X fails vulnerability audit VA-2026, what are the compliance consequences in Europe?",
        "What battery chemistry and thermal safety systems are used in Titan PowerPack P-500?",
        "What USMCA Regional Value Content percentage does Apex Assembly Plant contribute to ADU-7?",
        "What ESG protocol applies if refiner Cobalt Core Metals fails its RMI re-certification by June 2026?"
    ]
    
    selected_sample = st.selectbox("💡 Demo Prompts", options=sample_questions)
    
    default_query = selected_sample if selected_sample != "Select a sample question..." else ""
    user_query = st.text_input("Enter your question:", value=default_query, placeholder="e.g. How does Supplier X's regional policy affect ADU-7 compliance?")

    if st.button("🚀 Run GraphRAG Query", type="primary", use_container_width=True) and user_query:
        with st.spinner("Traversing Knowledge Graph & Dense Vector Index..."):
            start_t = time.time()
            
            # Execute Query via API or Direct Engines
            if use_api:
                payload = {
                    "query": user_query,
                    "mode": retrieval_mode,
                    "top_k_vector": top_k,
                    "hops": hops,
                    "enable_crag": enable_crag
                }
                res = requests.post(f"{API_BASE_URL}/query", json=payload).json()
                answer = res["answer"]
                provenance = res["provenance"]
                exec_time = res["execution_time_ms"]
            else:
                eng = engines
                v_chunks, g_triplets, matched = eng["retriever"].retrieve(
                    user_query, mode=retrieval_mode, top_k=top_k, hops=hops
                )
                context_str = eng["retriever"].format_context(v_chunks, g_triplets)
                crag_res = None
                if enable_crag:
                    crag_res = eng["crag"].evaluate_context(user_query, context_str)
                    if crag_res.web_fallback_triggered:
                        snippets = eng["crag"].search_web_fallback(user_query)
                        crag_res.web_snippets = snippets
                        context_str += f"\n\n### WEB CONTEXT:\n" + "\n".join(snippets)

                answer = eng["retriever"].generate_answer(user_query, context_str)
                exec_time = round((time.time() - start_t) * 1000.0, 2)
                
                provenance = {
                    "vector_chunks": [c.dict() for c in v_chunks],
                    "graph_triplets": [t.dict() for t in g_triplets],
                    "retrieved_entities": matched,
                    "crag_evaluation": crag_res.dict() if crag_res else None
                }

        # Response Banner
        st.markdown("### 💡 Synthesized Answer")
        st.info(answer)

        # CRAG Badge & Metrics Row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Execution Latency", f"{exec_time} ms")
        with col_m2:
            st.metric("Retrieved Chunks", len(provenance.get("vector_chunks", [])))
        with col_m3:
            st.metric("Sub-Graph Triplets", len(provenance.get("graph_triplets", [])))
        with col_m4:
            crag_eval = provenance.get("crag_evaluation")
            if crag_eval:
                conf = crag_eval.get("confidence_score", 0.0)
                grade = crag_eval.get("grade", "Correct")
                badge_cls = f"crag-badge-{grade.lower()}"
                st.markdown(f"**CRAG Grade**: <span class='{badge_cls}'>{grade} ({int(conf*100)}%)</span>", unsafe_allow_html=True)
                if crag_eval.get("web_fallback_triggered"):
                    st.caption("🌐 External Web Search Fallback Triggered")

        st.divider()

        # Two Column Layout: Sub-Graph Visualizer & Citation Cards
        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            st.markdown("### 🕸️ Interactive Sub-Graph Neighborhood")
            triplets_list = provenance.get("graph_triplets", [])
            if triplets_list:
                if use_api:
                    pyvis_url = f"{API_BASE_URL}/graph/pyvis?query={requests.utils.quote(user_query)}"
                    components.html(f'<iframe src="{pyvis_url}" width="100%" height="450" style="border:none;"></iframe>', height=450)
                elif engines:
                    from src.models import GraphTriplet
                    t_objs = [GraphTriplet(**t) for t in triplets_list]
                    html_content = engines["graph"].generate_pyvis_html(t_objs, highlight_nodes=provenance.get("retrieved_entities"))
                    components.html(html_content, height=460)
            else:
                st.warning("No Knowledge Graph relations were triggered for this query.")

        with right_col:
            st.markdown("### 📄 Cited Provenance Cards")
            v_chunks = provenance.get("vector_chunks", [])
            for i, chunk in enumerate(v_chunks):
                with st.expander(f"📌 [{i+1}] {chunk.get('doc_title')} (Similarity: {int(chunk.get('similarity_score', 0)*100)}%)"):
                    st.write(chunk.get("text"))

            if provenance.get("graph_triplets"):
                with st.expander("🔗 Explicit Knowledge Triplet Facts"):
                    for t in provenance.get("graph_triplets", []):
                        st.markdown(f"- **{t.get('subject')}** `--[{t.get('relation')}]-->` **{t.get('object')}**")

# -----------------------------------------------------------------------------
# TAB 2: KNOWLEDGE BASE INGESTION
# -----------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="main-title">Knowledge Base Ingestion & Graph Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload enterprise documents, extract triplets, and query Knowledge Graph stats</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### ➕ Upload Custom Document")
        doc_title = st.text_input("Document Title", placeholder="e.g. Supplier Q3 Security Audit")
        doc_category = st.selectbox("Category", ["Supply Chain", "Regulatory Compliance", "Product Engineering", "ESG Compliance", "General"])
        doc_content = st.text_area("Document Content", height=200, placeholder="Paste enterprise report, compliance text, or specs here...")

        if st.button("Ingest Document", type="primary"):
            if doc_title and doc_content:
                new_doc = {
                    "id": f"DOC-CUSTOM-{int(time.time())}",
                    "title": doc_title,
                    "category": doc_category,
                    "content": doc_content
                }
                
                if use_api:
                    r = requests.post(f"{API_BASE_URL}/ingest", json=[new_doc])
                    res = r.json()
                    st.success(f"Successfully ingested! Created {res['chunks_created']} vector chunks and {res['nodes_added']} graph nodes.")
                elif engines:
                    from src.models import Document
                    doc_obj = Document(**new_doc)
                    chunks_cnt = engines["vector"].ingest_documents([doc_obj])
                    triplets = engines["llm"].extract_triplets(doc_obj.content, doc_obj.id)
                    engines["graph"].add_triplets(triplets)
                    st.success(f"Successfully ingested locally! Added {len(triplets)} relational triplets to Knowledge Graph.")
                st.rerun()

    with c2:
        st.markdown("### 🌐 Whole Graph Explorer")
        if use_api:
            pyvis_full_url = f"{API_BASE_URL}/graph/pyvis"
            components.html(f'<iframe src="{pyvis_full_url}" width="100%" height="450" style="border:none;"></iframe>', height=450)
        elif engines:
            from src.models import GraphTriplet
            all_edges = list(engines["graph"].graph.edges(data=True))[:40]
            sample_t = [GraphTriplet(subject=str(u), relation=d.get("relation", "connected"), object=str(v)) for u, v, d in all_edges]
            html_full = engines["graph"].generate_pyvis_html(sample_t)
            components.html(html_full, height=450)

        # Graph Export Toolbar
        st.markdown("#### 📥 Export Knowledge Graph")
        exp_col1, exp_col2 = st.columns(2)
        
        # Safely fetch graphml and cypher from API or direct engine
        g_xml = ""
        g_cypher = ""
        try:
            if use_api:
                g_xml = requests.get(f"{API_BASE_URL}/graph/export/graphml").text
                g_cypher = requests.get(f"{API_BASE_URL}/graph/export/cypher").text
            elif engines and "graph" in engines:
                g_obj = engines["graph"]
                if hasattr(g_obj, "export_graphml"):
                    g_xml = g_obj.export_graphml()
                    g_cypher = g_obj.export_cypher()
                else:
                    from src.graph_engine import GraphEngine
                    temp_g = GraphEngine()
                    g_xml = temp_g.export_graphml()
                    g_cypher = temp_g.export_cypher()
        except Exception as e:
            st.error(f"Export error: {e}")

        with exp_col1:
            st.download_button("💾 Download GraphML (Gephi)", data=g_xml, file_name="knowledge_graph.graphml", mime="application/xml", use_container_width=True)
        with exp_col2:
            st.download_button("💾 Download Neo4j Cypher (.cypher)", data=g_cypher, file_name="knowledge_graph.cypher", mime="text/plain", use_container_width=True)



# -----------------------------------------------------------------------------
# TAB 3: BENCHMARK & RAGAS ANALYTICS
# -----------------------------------------------------------------------------
with tab3:
    st.markdown('<div class="main-title">RAGAS & Offline Benchmark Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Automated performance evaluation: Vector-Only Baseline vs GraphRAG (Hybrid)</div>', unsafe_allow_html=True)

    num_eval_questions = st.slider("Number of Benchmark Questions to Evaluate", min_value=5, max_value=35, value=15)

    if st.button("📊 Run Full Offline Benchmark Evaluation", type="primary", use_container_width=True):
        with st.spinner(f"Evaluating {num_eval_questions} multi-hop questions across Vector-Only and GraphRAG modes..."):
            if use_api:
                r = requests.get(f"{API_BASE_URL}/eval?max_questions={num_eval_questions}")
                results = r.json()["benchmark_results"]
            elif engines:
                results = engines["evaluator"].run_full_comparison(max_questions=num_eval_questions)

            st.session_state["eval_results"] = results

    if "eval_results" in st.session_state:
        results = st.session_state["eval_results"]
        vec_m = results["vector_only"]
        graph_m = results["graphrag_hybrid"]

        if hasattr(vec_m, "dict"):
            vec_m = vec_m.dict()
        if hasattr(graph_m, "dict"):
            graph_m = graph_m.dict()

        st.divider()
        st.markdown("### 📈 Key Benchmark Comparison Metrics")


        m1, m2, m3, m4 = st.columns(4)
        with m1:
            diff_recall = round(graph_m["context_recall"] - vec_m["context_recall"], 2)
            st.metric("Context Recall", f"{int(graph_m['context_recall']*100)}%", delta=f"{int(diff_recall*100)}% vs Vector")
        with m2:
            diff_faith = round(graph_m["answer_faithfulness"] - vec_m["answer_faithfulness"], 2)
            st.metric("Answer Faithfulness", f"{int(graph_m['answer_faithfulness']*100)}%", delta=f"{int(diff_faith*100)}% vs Vector")
        with m3:
            diff_halluc = round(vec_m["hallucination_score"] - graph_m["hallucination_score"], 2)
            st.metric("Hallucination Score", f"{graph_m['hallucination_score']}", delta=f"-{diff_halluc} (Lower is Better)")
        with m4:
            st.metric("Avg Latency", f"{graph_m['avg_latency_ms']} ms", delta=f"{vec_m['avg_latency_ms']} ms Vector")

        # Plotly Bar Chart Comparison
        categories = ["Context Recall (%)", "Answer Faithfulness (%)", "Hallucination Score (0-1)"]
        vec_vals = [vec_m["context_recall"] * 100, vec_m["answer_faithfulness"] * 100, vec_m["hallucination_score"]]
        graph_vals = [graph_m["context_recall"] * 100, graph_m["answer_faithfulness"] * 100, graph_m["hallucination_score"]]

        fig = go.Figure(data=[
            go.Bar(name='Vector-Only Baseline', x=categories, y=vec_vals, marker_color='#64748B'),
            go.Bar(name='GraphRAG Hybrid', x=categories, y=graph_vals, marker_color='#3B82F6')
        ])
        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#F8FAFC"),
            title="Ablation Benchmark: Vector-Only vs. GraphRAG (Hybrid)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Question Level Table Breakdown
        st.markdown("### 📋 Individual Question Evaluation Log")
        details_data = []
        for d in graph_m.get("details", []):
            details_data.append({
                "Q-ID": d.get("question_id"),
                "Question": d.get("question"),
                "Complexity": d.get("complexity"),
                "Context Recall": f"{int(d.get('context_recall', 0)*100)}%",
                "Faithfulness": f"{int(d.get('answer_faithfulness', 0)*100)}%",
                "Latency (ms)": d.get("latency_ms")
            })
        st.dataframe(details_data, use_container_width=True)
