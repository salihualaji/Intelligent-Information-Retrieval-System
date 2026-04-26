import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from retrieval_models.bm25_retrieval import TFIDFRetrieval
from retrieval_models.sbert_retrieval import SBERTRetrieval


docs = pd.read_csv("data/ir_documents.csv")
queries = pd.read_csv("data/ir_queries.csv")
relevance = pd.read_csv("data/ir_relevance.csv")


SBERTRetrieval_model = SBERTRetrieval(docs)
SBERTRetrieval_model = SBERTRetrieval(docs)


results = []


for _, row in queries.iterrows():

    query_id = row["query_id"]
    query = row["query"]

    relevant_docs = relevance[relevance.query_id == query_id]["doc_id"].tolist()

    SBERTRetrieval_results = SBERTRetrieval_model.search(query)
    SBERTRetrieval_results = SBERTRetrieval_model.search(query)


    # Create binary vectors
    y_true = [1 if doc in relevant_docs else 0 for doc in docs.doc_id]

    SBERTRetrieval_pred = [1 if doc in SBERTRetrieval_results else 0 for doc in docs.doc_id]
    SBERTRetrieval_pred = [1 if doc in SBERTRetrieval_results else 0 for doc in docs.doc_id]


    tfidf_precision = precision_score(y_true, SBERTRetrieval_pred)
    tfidf_recall = recall_score(y_true, SBERTRetrieval_pred)
    tfidf_f1 = f1_score(y_true, SBERTRetrieval_pred)


    SBERTRetrieval_precision = precision_score(y_true, SBERTRetrieval_pred)
    SBERTRetrieval_recall = recall_score(y_true, SBERTRetrieval_pred)
    SBERTRetrieval_f1 = f1_score(y_true, SBERTRetrieval_pred)


    results.append({
        "Query": query,
        "SBERT Precision": SBERTRetrieval_precision,
        "SBERT Recall": SBERTRetrieval_recall,
        "SBERT F1": SBERTRetrieval_f1,
        "SBERT Precision": SBERTRetrieval_precision,
        "SBERT Recall": SBERTRetrieval_recall,
        "SBERT F1": SBERTRetrieval_f1
    })


df = pd.DataFrame(results)

print(df)

df.to_csv("evaluation_results.csv", index=False)