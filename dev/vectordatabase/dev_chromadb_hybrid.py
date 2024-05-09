""" Try different retrieval methods """

from pathlib import Path
import shutil
import uuid
import sqlite3

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import numpy as np

from sentence_transformers import SentenceTransformer



nomic_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
)


class NomicEmbedText(EmbeddingFunction):
    def __call__(self, doc: Documents) -> Embeddings:
        embeddings = nomic_model.encode(doc)
        return list(embeddings.tolist())

class HybridSearch:
    
    def __init__(self, path):
        self._chromadb_client = chromadb.PersistentClient(path=str(path))
        self._collection = self._chromadb_client.get_or_create_collection("base", embedding_function=NomicEmbedText())
        self._dbase = Path(self._chromadb_client._identifier) / "chroma.sqlite3"
        self._conn = sqlite3.connect(self._dbase)
        self._init_fts()

    def _init_fts(self):
        c = self._conn.cursor()
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING FTS5(document_id, content)")
        self._conn.commit()
        c.close()

    def add(self, documents, metadatas, ids):
        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        c = self._conn.cursor()
        for id_, document in zip(ids, documents):
            # possible preprocessing like lemmatization
            c.execute("INSERT INTO fts VALUES (?, ?)", (id_, document))
        self._conn.commit()
        c.close()

    def query_bm25(self, query, k):
        c = self._conn.cursor()
        sqlquery = "SELECT *, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?"
        c.execute(sqlquery, (query, k))
        resp = c.fetchall()
        ids = []
        documents = []
        scores = []
        for respi in resp:
            ids.append(respi[0])
            documents.append(respi[1])
            scores.append(respi[2])
        return {"ids": ids, "scores": scores, "documents": documents}
        
    def query_embedding(self, query, k):
        result = self._collection.query(query_texts=[query], n_results=k)
        return {"ids": result["ids"][0], "scores": [1/dist for dist in result["distances"][0]], "documents": result["documents"][0]}

    def query_hybrid(self, query, top_k=2, bm25_weight=0.5, embedding_weight=0.5):
        result_bm25 = self.query_bm25(query, 3)
        result_embedding = self.query_embedding(query, 3)

        # Extract scores and document IDs
        bm25_ids, bm25_scores = np.array(result_bm25["ids"]), np.array(result_bm25["scores"])
        embedding_ids, embedding_scores = np.array(result_embedding["ids"]), np.array(result_embedding["scores"])

        # Normalize scores by their sum
        bm25_scores_norm = bm25_scores / np.sum(bm25_scores)
        embedding_scores_norm = embedding_scores / np.sum(embedding_scores)

        # Map scores to dictionaries for easy lookup
        bm25_dict = dict(zip(bm25_ids, bm25_scores_norm))
        embedding_dict = dict(zip(embedding_ids, embedding_scores_norm))

        # Union of all document IDs
        all_doc_ids = set(bm25_ids).union(embedding_ids)

        # Initialize combined scores array
        combined_scores = []

        # Compute combined scores for each document
        for doc_id in all_doc_ids:
            bm25_score = bm25_dict.get(doc_id, 0)  # Default to 0 if not present
            embedding_score = embedding_dict.get(doc_id, 0)  # Default to 0 if not present
            combined_score = (bm25_weight * bm25_score) + (embedding_weight * embedding_score)
            combined_scores.append((doc_id, combined_score))

        
        top_doc_indices = np.argsort(combined_scores)[::-1][:top_k]
        top_doc_ids = all_doc_ids[top_doc_indices]
        #return [(documents[idx], combined_scores[idx]) for idx in top_doc_indices]



example_data = [
    {
        "name": "Skye",
        "type": "Person",
        "summary": "Bayonette's fixer who provides her with a job opportunity. She carries various documents relating to the run.",
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
        "summary": "A large sword belonging to the contractor, a powerful elf, that has been stolen. With it a significant portion of the family deeds were lost.",
    },
    {
        "name": "The Cloister",
        "type": "Location",
        "summary": "A high class restaurant for the elite that is located on an artificial mountain several kilometers away from Seattle.",
    },
]

# we need to delete all data when we want to create a new collection
# for hybrid search we can only use one collection since chromadb does not
# distinguish its FTS table between different collections
path = Path(__file__).parent / ".chromadb_hybrid"
shutil.rmtree(path)
hybrid_search = HybridSearch(path)

# since we want to search the whole document with fulltext-search we fuse all json
# fields
hybrid_search.add(
    documents=[f"{data["name"]} ({data["type"]}): {data["summary"]}" for data in example_data],
    metadatas=[
        {"name": data["name"], "type": data["type"], "summary": data["summary"]}
        for data in example_data
    ],
    ids=[str(uuid.uuid4()) for _ in example_data],
)


result = hybrid_search.query_embedding("documents", 2)
print("Embedding Result:", result)

result = hybrid_search.query_bm25("documents", 2)
print("BM25 Result:", result)

result = hybrid_search.query_hybrid("documents", 2)
