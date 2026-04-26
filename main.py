import pandas as pd

from retrieval_models.bm25_retrieval import BM25Retrieval
from retrieval_models.sbert_retrieval import SBERTRetrieval

from evaluation.evaluation_metrics import precision_at_k, recall


# ==============================
# Load Dataset
# ==============================

docs = pd.read_csv("data/ir_documents.csv")
queries = pd.read_csv("data/ir_queries.csv")  # must include query_type column
relevance = pd.read_csv("data/ir_relevance.csv")


# ==============================
# Initialize Models
# ==============================

bm25_model = BM25Retrieval(docs)
sbert_model = SBERTRetrieval(docs)


# ==============================
# Store Results
# ==============================

results = []


# ==============================
# Run Experiment
# ==============================

for _, row in queries.iterrows():

    query = row["query"]
    qid = row["query_id"]


    query_type = row.get("query_type", "normal")

    relevant_docs = relevance[relevance.query_id == qid]["doc_id"].tolist()

    # Retrieve documents
    bm25_results = bm25_model.search(query)
    sbert_results = sbert_model.search(query)

    # Compute Metrics
    bm25_precision = precision_at_k(bm25_results, relevant_docs, 5)
    bm25_recall = recall(bm25_results, relevant_docs)

    sbert_precision = precision_at_k(sbert_results, relevant_docs, 5)
    sbert_recall = recall(sbert_results, relevant_docs)

    # Store results
    results.append({
        "query_id": qid,
        "query": query,
        "query_type": query_type,

        "bm25_precision": bm25_precision,
        "bm25_recall": bm25_recall,

        "sbert_precision": sbert_precision,
        "sbert_recall": sbert_recall
    })


# ==============================
# Save Results
# ==============================

results_df = pd.DataFrame(results)

results_df.to_csv("results/experiment_results.csv", index=False)


# ==============================
# Overall Performance
# ==============================

print("\n===== Overall System Performance =====\n")

print("BM25 Precision:", results_df["bm25_precision"].mean())
print("BM25 Recall:", results_df["bm25_recall"].mean())

print("\nSBERT Precision:", results_df["sbert_precision"].mean())
print("SBERT Recall:", results_df["sbert_recall"].mean())


# ==============================
# Query-Type Analysis
# ==============================

print("\n===== Performance by Query Type =====\n")

grouped = results_df.groupby("query_type").mean(numeric_only=True)

print(grouped)