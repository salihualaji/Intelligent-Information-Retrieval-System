import pandas as pd 
import streamlit as st

from retrieval_models.bm25_retrieval import BM25Retrieval
from retrieval_models.sbert_retrieval import SBERTRetrieval


# PAGE CONFIG
st.set_page_config(
    page_title="AI Information Retrieval System",
    layout="wide"
)

# CUSTOM STYLES
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.hero {
    padding:40px;
    border-radius:12px;
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
    margin-bottom:30px;
}

.hero h1{
    font-size:48px;
}

.hero p{
    font-size:18px;
    opacity:0.9;
}

.result-card{
    background:white;
    padding:20px;
    border-radius:12px;
    margin-bottom:20px;
    border:1px solid #eaeaea;
    transition:0.2s;
}

.result-card:hover{
    box-shadow:0 8px 20px rgba(0,0,0,0.1);
}

.search-container{
    padding:30px;
    border-radius:10px;
    background:#f7f9fc;
    margin-bottom:30px;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:10px;
    text-align:center;
    border:1px solid #eee;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# SIDEBAR (DATASET SELECTOR FIRST)
# ==============================
st.sidebar.title("System Dashboard")

dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["University FAQs", "UK Universities"]
)

# ==============================
# LOAD DATA BASED ON SELECTION
# ==============================

if dataset_option == "University FAQs":
    docs = pd.read_csv("data/ir_documents.csv")

elif dataset_option == "UK Universities":
    docs = pd.read_csv("data/uk_universities_documents.csv")


# ==============================
# INITIALIZE MODELS (AFTER DATA LOAD)
# ==============================

bm25_model = BM25Retrieval(docs)
sbert_model = SBERTRetrieval(docs)


# ==============================
# SIDEBAR METRICS (DYNAMIC)
# ==============================

st.sidebar.metric("Documents Indexed", len(docs))
st.sidebar.metric("Models Available", "2")
st.sidebar.metric("System Status", "Active")

st.sidebar.markdown("---")
st.sidebar.success(f"Dataset: {dataset_option}")
st.sidebar.info("Baseline: BM25 | Intelligent Model: SBERT")


# ==============================
# HERO SECTION
# ==============================

st.markdown("""
<div class="hero">
<h1> Intelligent Information Retrieval System</h1>
<p>
A hybrid AI-powered search system designed to compare traditional 
and intelligent retrieval techniques for improved information access 
across multiple datasets.
</p>
</div>
""", unsafe_allow_html=True)


# ==============================
# SEARCH PANEL
# ==============================

st.markdown('<div class="search-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([3,1,1])

with col1:
    query = st.text_input("Enter your search query")

with col2:
    model_choice = st.selectbox(
        "Retrieval Model",
        ["BM25 (Baseline)", "SBERT (Semantic AI)"]
    )

with col3:
    query_type = st.selectbox(
        "Query Type",
        ["Normal", "Short Query", "Synonym", "Paraphrased"]
    )

search = st.button("🔍 Search")

st.markdown("</div>", unsafe_allow_html=True)


# ==============================
# RESULTS
# ==============================

if search and query:

    modified_query = query

    if query_type == "Short Query":
        modified_query = " ".join(query.split()[:2])

    elif query_type == "Synonym":
        synonyms = {
            "reset": "change",
            "password": "credentials",
            "course": "class",
            "registration": "enrollment",
            "exam": "test",
            "schedule": "timetable"
        }
        words = query.lower().split()
        modified_query = " ".join([synonyms.get(w, w) for w in words])

    elif query_type == "Paraphrased":
        modified_query = f"Explain how to {query.lower()}"

    st.info(f"""
    Original Query: {query}  
    Modified Query: {modified_query}  
    Query Type: {query_type}
    """)

    # Retrieval
    if model_choice == "BM25 (Baseline)":
        results = bm25_model.search(modified_query)
    else:
        results = sbert_model.search(modified_query)

    st.subheader(f"Top {len(results)} Search Results")

    for rank, doc_id in enumerate(results, start=1):
        doc_match = docs[docs.doc_id == doc_id]

        if doc_match.empty:
            continue

        doc = doc_match.iloc[0]

        st.markdown(f"""
        <div class="result-card">
        <h3>#{rank} {doc['title']}</h3>
        <p>{doc['content']}</p>
        </div>
        """, unsafe_allow_html=True)