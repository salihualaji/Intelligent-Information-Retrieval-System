from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SBERTRetrieval:
    def __init__(self, documents):
        self.documents = documents
        
        # Load pre-trained SBERT model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode all documents into embeddings
        self.doc_embeddings = self.model.encode(
            documents['content'].tolist()
        )

    def search(self, query):
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Compute similarity
        similarities = cosine_similarity(
            query_embedding, 
            self.doc_embeddings
        ).flatten()
        
        # Rank documents by similarity
        ranked_indices = similarities.argsort()[::-1]
        
        # STEP 2: Convert indices → actual document IDs
        top_docs = self.documents.iloc[ranked_indices[:5]]
        
        return top_docs["doc_id"].tolist()