from rank_bm25 import BM25Okapi
import re

class BM25Retrieval:
    def __init__(self, documents):
        self.documents = documents
        
        # Preprocess documents (lowercase + tokenization)
        self.tokenized_docs = [
            re.findall(r'\w+', doc.lower()) 
            for doc in documents['content']
        ]
        
        # Initialize BM25 model
        self.model = BM25Okapi(self.tokenized_docs)

    def search(self, query):
        # Preprocess query
        tokenized_query = re.findall(r'\w+', query.lower())
        
        # Get BM25 scores
        scores = self.model.get_scores(tokenized_query)
        
        # Rank documents by score
        ranked_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )
        
        # Convert indices → actual document IDs
        top_docs = self.documents.iloc[ranked_indices[:5]]
        
        return top_docs["doc_id"].tolist()