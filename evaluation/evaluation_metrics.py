def precision_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    relevant_set = set(relevant)

    retrieved_relevant = len([doc for doc in retrieved_k if doc in relevant_set])

    return retrieved_relevant / k


def recall(retrieved, relevant):

    relevant_set = set(relevant)

    retrieved_relevant = len([doc for doc in retrieved if doc in relevant_set])

    return retrieved_relevant / len(relevant_set)