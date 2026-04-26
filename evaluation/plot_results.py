import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Load results
df = pd.read_csv("results/experiment_results.csv")

# ==============================
# Overall Precision & Recall
# ==============================

bm25_precision = df["bm25_precision"].mean()
sbert_precision = df["sbert_precision"].mean()

bm25_recall = df["bm25_recall"].mean()
sbert_recall = df["sbert_recall"].mean()

# Precision Plot
plt.figure()
models = ["BM25", "SBERT"]
precision_scores = [bm25_precision, sbert_precision]

plt.bar(models, precision_scores)
plt.title("Precision Comparison")
plt.xlabel("Models")
plt.ylabel("Precision")

plt.savefig("results/precision_comparison.png")
plt.close()

# Recall Plot
plt.figure()
recall_scores = [bm25_recall, sbert_recall]

plt.bar(models, recall_scores)
plt.title("Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Recall")

plt.savefig("results/recall_comparison.png")
plt.close()


# ==============================
# Query-Type Analysis
# ==============================

grouped = df.groupby("query_type").mean(numeric_only=True)

query_types = grouped.index.tolist()

# BM25 by query type
plt.figure()
plt.bar(query_types, grouped["bm25_precision"])
plt.title("BM25 Precision by Query Type")
plt.xlabel("Query Type")
plt.ylabel("Precision")

plt.savefig("results/bm25_query_type.png")
plt.close()

# SBERT by query type
plt.figure()
plt.bar(query_types, grouped["sbert_precision"])
plt.title("SBERT Precision by Query Type")
plt.xlabel("Query Type")
plt.ylabel("Precision")

plt.savefig("results/sbert_query_type.png")
plt.close()


# ==============================
# Combined Comparison (BEST ONE)
# ==============================

plt.figure()

x = range(len(query_types))

plt.plot(x, grouped["bm25_precision"], marker='o', label="BM25")
plt.plot(x, grouped["sbert_precision"], marker='o', label="SBERT")

plt.xticks(x, query_types)
plt.title("Model Performance Across Query Types")
plt.xlabel("Query Type")
plt.ylabel("Precision")
plt.legend()

plt.savefig("results/model_comparison_query_types.png")
plt.close()


print("\ Graphs generated successfully in /results folder")