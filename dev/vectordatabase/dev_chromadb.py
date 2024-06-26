""" Try different retrieval methods """

import uuid

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from sentence_transformers import SentenceTransformer


nomic_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
)


class NomicEmbedText(EmbeddingFunction):
    def __call__(self, doc: Documents) -> Embeddings:
        embeddings = nomic_model.encode(doc)
        return list(embeddings.tolist())


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

# embedding is calculated on documents. In that case on the content of the "name" field
# all other data is metadata and hence not considered
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
