import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Algorithms dealing with retrieving information from large datasets.",
    "BM25 is a popular ranking function used in information retrieval.",
    "TF-IDF is a widely used weighting scheme in information retrieval.",
    "A banana is a yellow fruit.",
]

# Initialize the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode the documents to get their embeddings
document_embeddings = model.encode(documents)


def semantic_search(query):
    # Encode the query to get its embedding
    query_embedding = model.encode([query])

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, document_embeddings)

    # Get the sorted list of document indices based on their similarity scores
    sorted_doc_indices = np.argsort(similarities[0])[::-1]

    # Print the results
    print("Semantic Search Results:")
    for idx in sorted_doc_indices:
        print(
            f"Doc {idx + 1}: Similarity Score: {similarities[0][idx]:.4f}, Document: '{documents[idx]}'"
        )


# Example query
# search_query = "web search"
search_query = "information retrieval"
semantic_search(search_query)
