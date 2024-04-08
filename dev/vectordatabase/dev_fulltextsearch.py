from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np


def normalize(doc):
    return doc.lower().replace(".", "")


# Sample documents
documents = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Algorithms dealing with retrieving information from large datasets.",
    "BM25 is a popular ranking function used in information retrieval.",
    "TF-IDF is a widely used weighting scheme in information retrieval.",
    "A banana is a yellow fruit.",
]

# Preprocess documents (remove punctuation, lowercase, etc.)
# Might to be more sophisticated for real world application
preprocessed_docs = [normalize(doc) for doc in documents]
# rank_bm25 implementation needs tokenized input
tokenized_docs = [doc.split(" ") for doc in preprocessed_docs]

# TF-IDF representation
tfidf_vectorizer = TfidfVectorizer()
tfidf_doc_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)

# BM25 representation
bm25_indexer = BM25Okapi(tokenized_docs)


def search(query):

    query_preprocessed = normalize(query)

    # Calculate TF-IDF scores
    query_tfidf_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = np.dot(query_tfidf_vector, tfidf_doc_matrix.T).todense()
    cosine_similarities = np.asarray(cosine_similarities)[0]

    # Calculate BM25 scores
    bm25_scores = bm25_indexer.get_scores(query_preprocessed.split(" "))

    combined_scores = sorted(
        zip(cosine_similarities, bm25_scores, documents),
        key=lambda x: x[0] + x[1],
        reverse=True,
    )

    # Print the results
    print("Ranking based on TF-IDF and BM25:")
    for idx, (tfidf_score, bm25_score, document) in enumerate(combined_scores):
        print(
            f"Doc {idx + 1}: TF-IDF Score: {tfidf_score:.4f}, BM25 Score: {bm25_score:.4f}, Document: '{document}'"
        )


# Example query
search_query = "web search"
# search_query = "information retrieval"
search(search_query)
