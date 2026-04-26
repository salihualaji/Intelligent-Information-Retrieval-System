import pandas as pd

class BooleanRetrieval:

    def __init__(self, documents):
        self.documents = documents

    def search(self, query):

        query_terms = query.lower().split()

        results = []

        for index, row in self.documents.iterrows():

            text = (row['title'] + " " + row['content']).lower()

            if all(term in text for term in query_terms):
                results.append(row['doc_id'])

        return results