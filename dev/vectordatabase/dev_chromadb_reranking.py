""" Try different retrieval methods """

from pathlib import Path
import uuid

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

# doesnt work on windows
# from ragatouille import RAGPretrainedModel
# RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")


nomic_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
)


class NomicEmbedText(EmbeddingFunction):
    def __call__(self, doc: Documents) -> Embeddings:
        embeddings = nomic_model.encode(doc)
        return list(embeddings.tolist())


# class RagatouilleEmbed(EmbeddingFunction):
#     def __call__(self, input: Documents) -> Embeddings:
#         embeddings = rag_model.encode(input)
#         return list(embeddings.tolist())


def maxsim(query_embedding, document_embedding):
    # Expand dimensions for broadcasting
    # Query: [batch_size, query_length, embedding_size]
    #   -> [batch_size, query_length, 1, embedding_size]
    # Document: [batch_size, doc_length, embedding_size]
    #   -> [batch_size, 1, doc_length, embedding_size]
    expanded_query = query_embedding.unsqueeze(2)
    expanded_doc = document_embedding.unsqueeze(1)

    # Compute cosine similarity across the embedding dimension\n",
    sim_matrix = torch.nn.functional.cosine_similarity(
        expanded_query, expanded_doc, dim=-1
    )

    # Take the maximum similarity for each query token (across all document tokens)
    # sim_matrix shape: [batch_size, query_length, doc_length]
    max_sim_scores, _ = torch.max(sim_matrix, dim=2)

    # Average these maximum scores across all query tokens
    avg_max_sim = torch.mean(max_sim_scores, dim=1)
    return avg_max_sim


def colbert_rerank(documents: list[str], query: str):
    scores = []
    query_encoding = tokenizer(query, return_tensors="pt")
    query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)

    for doc in documents:
        doc_encoding = tokenizer(
            doc, return_tensors="pt", truncation=True, max_length=512
        )
        doc_embedding = model(**doc_encoding).last_hidden_state

        # Calculate MaxSim score
        score = maxsim(query_embedding.unsqueeze(0), doc_embedding)
        scores.append(
            {
                "score": score.item(),
                "document": doc,
            }
        )

    return scores


print(colbert_rerank(["banana", "apple", "gun"], "pear"))


example_data = [
    {
        "name": "Skye",
        "type": "Person",
        "summary": "Bayonette's fixer who provides her with a job opportunity.",
    },
    {
        "name": "The Client",
        "type": "Person",
        "summary": "A metahuman rights activist who hires Bayonette to shut down a biological weapon targeting metahumans.",
    },
    {
        "name": "Shutdown Target",
        "type": "Location",
        "summary": "A research facility on the outskirts of Denver that develops a biological weapon targeting metahumans. Known as the 'Twisted Tetrahedron Lab.'",
    },
    {
        "name": "Metamorphosis",
        "type": "Item",
        "summary": "The codename for the biological weapon that Bayonette needs to obtain information on.",
    },
    {
        "name": "Replacement Project",
        "type": "Organization",
        "summary": "A shadowy group behind the development of the 'Metamorphosis' biological weapon. Specialized in genetic manipulation and bioweapons.",
    },
    {
        "name": "Family Heirloom",
        "type": "Item",
        "summary": "A large sword belonging to the contractor, a powerful elf, that has been stolen.",
    },
    {
        "name": "The Cloister",
        "type": "Location",
        "summary": "A high class restaurant for the elite that is located on an artificial mountain several kilometers away from Seattle.",
    },
]

client = chromadb.Client()

try:
    client.delete_collection(name="entities")
except ValueError:
    pass

collection = client.create_collection("entities", embedding_function=NomicEmbedText())

collection.add(
    documents=[data["name"] for data in example_data],
    metadatas=[
        {"name": data["name"], "type": data["type"], "summary": data["summary"]}
        for data in example_data
    ],
    ids=[str(uuid.uuid4()) for _ in example_data],
)


result = collection.query(query_texts=["employer"], n_results=1)

print(result)
