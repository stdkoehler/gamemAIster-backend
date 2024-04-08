from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


documents = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Algorithms dealing with retrieving information from large datasets.",
    "BM25 is a popular ranking function used in information retrieval.",
    "TF-IDF is a widely used weighting scheme in information retrieval.",
    "A banana is a yellow fruit.",
    "BM25 is a ranking function use in retrieval for web search.",
]


def normalize(doc):
    return doc.lower().replace(".", "")


# Preprocess documents for BM25
tokenized_corpus = [normalize(doc).split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(documents)


def semantic_scoring(query):
    query_embedding = embedding_model.encode([query])
    scores = cosine_similarity(query_embedding, document_embeddings)[0]
    return scores


def bm25_scoring(query):
    query_tokens = normalize(query).split(" ")
    scores = bm25.get_scores(query_tokens)
    return scores


def hybrid_search(query, top_k=2, bm25_weight=0.5, semantic_weight=0.5):

    semantic_scores = np.array(semantic_scoring(query))
    bm25_scores = np.array(bm25_scoring(query))

    semantic_scores_norm = semantic_scores / np.sum(semantic_scores)
    bm25_scores_norm = bm25_scores / np.sum(bm25_scores)

    combined_scores = (bm25_weight * bm25_scores_norm) + (
        semantic_weight * semantic_scores_norm
    )

    top_doc_indices = np.argsort(combined_scores)[::-1][:top_k]
    return [(documents[idx], combined_scores[idx]) for idx in top_doc_indices]


# Example usage
question = "How does web search retrieval work?"
best_documents = hybrid_search(question)
print(f"Best document for question '{question}' is: '{best_documents[0]}'")


# import os
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from faiss import IndexFlatIP, faiss
# from collections import defaultdict
# from typing import List, Dict

# documents = [
#     "Natural language processing is a subfield of artificial intelligence.",
#     "Algorithms dealing with retrieving information from large datasets.",
#     "BM25 is a popular ranking function used in information retrieval.",
#     "TF-IDF is a widely used weighting scheme in information retrieval.",
#     "A banana is a yellow fruit.",
# ]

# # Load the pre-trained MiniLM model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load the BM25 index and term weights if you have them. Otherwise, create a new one.
# bm25_index = ...  # Your BM25 index
# bm25_term_weights = ...  # Your BM25 term weights


# def load_documents(file_path: str) -> List[str]:
#     with open(file_path, "r") as f:
#         documents = [doc for doc in json.load(f)]
#     return documents


# def create_bm25_index(documents: List[str]) -> Dict[str, float]:
#     # Implement your own BM25 index creation here
#     raise NotImplementedError


# def calculate_bm25_scores(
#     query: str, documents: List[str], index: Dict[str, float]
# ) -> List[float]:
#     # Implement your own BM25 scoring here
#     raise NotImplementedError


# def hybrid_search(
#     query: str, documents: List[str], top_k: int, relevance_feedback: List[str] = None
# ) -> List[Tuple[str, float]]:
#     # Encode the query and documents using the MiniLM model
#     query_embedding = model.encode(query, convert_to_tensor=True)[0]
#     document_embeddings = model.encode(documents, convert_to_tensor=True)

#     # Create a flat index for the MiniLM embeddings
#     index = IndexFlatIP(document_embeddings.shape[1])
#     index.add(document_embeddings)

#     # Perform an approximate nearest neighbors search with MiniLM
#     distances, indices = index.search(query_embedding, top_k)

#     # Calculate BM25 scores
#     if relevance_feedback:
#         relevance_feedback_embeddings = model.encode(
#             relevance_feedback, convert_to_tensor=True
#         )
#         adjusted_query_embedding = np.mean(
#             [query_embedding] + relevance_feedback_embeddings, axis=0
#         )
#     else:
#         adjusted_query_embedding = query_embedding

#     bm25_scores = calculate_bm25_scores(adjusted_query_embedding, documents, bm25_index)

#     # Combine MiniLM and BM25 scores
#     hybrid_scores = [
#         (documents[i], distances[i, 0] * bm25_scores[i]) for i in indices[0]
#     ]

#     # Sort the results by hybrid score
#     hybrid_scores.sort(key=lambda x: x[1], reverse=True)

#     return hybrid_scores[:top_k]


# # Load your documents from a JSON file
# documents = load_documents("documents.json")

# # Create a BM25 index if you don't have one
# bm25_index = create_bm25_index(documents)

# # Load BM25 term weights if you have them
# bm25_term_weights = load_bm25_term_weights("bm25_weights.json")

# query = "Your search query here"
# top_k = 10
