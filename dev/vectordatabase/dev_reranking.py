""" Try different retrieval methods """

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

documents = [
    "Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. It encompasses various tasks such as text classification, sentiment analysis, named entity recognition, and machine translation, aiming to enable computers to understand, interpret, and generate human language.",
    "Machine learning algorithms play a crucial role in analyzing and extracting valuable insights from vast datasets. These algorithms learn patterns and relationships from data, enabling automated decision-making and prediction in diverse domains such as healthcare, finance, marketing, and e-commerce.",
    "BM25, also known as Okapi BM25, is a probabilistic ranking function used in information retrieval to estimate the relevance of documents to a given search query. It considers factors such as term frequency, document length, and document frequency to score and rank documents, providing more accurate search results compared to traditional ranking methods.",
    "TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that evaluates how important a word is to a document in a collection or corpus. It calculates a weight for each term based on its frequency in the document and its rarity across the entire corpus, helping to identify significant terms and distinguish them from common ones.",
    "A banana is a long, curved fruit with a soft flesh and a yellow skin when ripe. Bananas are rich in potassium, vitamins, and antioxidants, making them a nutritious and convenient snack option. They are widely cultivated in tropical regions and consumed worldwide in various forms, including fresh, dried, and processed products.",
    "Deep learning models, such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs), have revolutionized the field of natural language processing. These models excel at capturing complex patterns and dependencies in sequential data, enabling tasks such as language translation, sentiment analysis, and speech recognition with unprecedented accuracy.",
    "The PageRank algorithm, developed by Larry Page and Sergey Brin, is a key component of Google's search engine, ranking web pages based on their importance and relevance. It considers both the number of links to a page and the quality of those links, providing users with authoritative and trustworthy search results.",
    "Information retrieval techniques encompass a wide range of methods, including keyword matching, vector space models, and semantic analysis. These techniques aim to retrieve relevant information from large collections of unstructured or semi-structured data sources, such as text documents, web pages, and multimedia content.",
    "Semantic search aims to understand the intent and contextual meaning behind user queries to provide more accurate search results. It goes beyond keyword matching to analyze the semantics of words and phrases, taking into account synonyms, related concepts, and user context to deliver highly relevant and personalized search results.",
    "An apple is a crunchy and juicy fruit that comes in various colors, including red, green, and yellow. Apples are a rich source of fiber, vitamins, and antioxidants, contributing to a healthy diet and promoting overall well-being. They are commonly eaten raw but are also used in cooking, baking, and beverage production.",
    "Named entity recognition (NER) is a subtask of information extraction that identifies and classifies named entities mentioned in unstructured text into predefined categories such as person names, organizations, and locations. NER systems utilize machine learning algorithms and linguistic features to detect and categorize named entities accurately.",
    "Latent semantic indexing (LSI) is a technique used in natural language processing to analyze relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. LSI enables semantic understanding and similarity comparison between documents, facilitating tasks such as document clustering and information retrieval.",
    "Document clustering groups similar documents together based on their content or features, facilitating exploration and organization of large document collections. Clustering algorithms partition documents into clusters such that documents within the same cluster are more similar to each other than to those in other clusters, aiding in document categorization and summarization.",
    "Information retrieval systems often employ relevance feedback mechanisms, where users provide feedback on the relevance of retrieved documents, which is then used to refine subsequent search results. Relevance feedback enhances the effectiveness of information retrieval systems by incorporating user preferences and judgments into the ranking process.",
    "Word embeddings, such as Word2Vec and GloVe, represent words as dense vectors in a continuous vector space, capturing semantic similarities between words based on their distributional properties. Word embeddings facilitate various natural language processing tasks, including word similarity calculation, text classification, and sentiment analysis.",
    "Named entity disambiguation resolves ambiguities in named entity mentions by identifying the correct entity reference based on contextual clues and knowledge bases. Disambiguation techniques disambiguate named entities by considering factors such as context, co-reference resolution, and entity linking to improve the accuracy of downstream NLP applications.",
    "Sentiment analysis aims to determine the emotional tone or attitude expressed in text, often used in applications such as social media monitoring and customer feedback analysis. Sentiment analysis algorithms classify text into positive, negative, or neutral sentiment categories, enabling organizations to understand public opinion and sentiment trends.",
    "Text summarization techniques condense large bodies of text into shorter versions while preserving the most important information and main points. Summarization methods can be extractive, selecting and rephrasing key sentences from the original text, or abstractive, generating new sentences that capture the essence of the text.",
    "Named entity linking connects named entity mentions in text to their corresponding entities in a knowledge base or database, enabling richer semantic understanding and integration with other data sources. Named entity linking algorithms disambiguate entity mentions and resolve references to entities by leveraging knowledge bases, entity catalogs, and semantic similarity measures.",
    "Collaborative filtering is a recommendation technique that predicts user preferences based on the preferences of similar users or items, commonly used in recommendation systems for movies, music, and products. Collaborative filtering algorithms analyze user-item interaction data to generate personalized recommendations, enhancing user satisfaction and engagement.",
    "Query expansion enhances search results by adding relevant terms or synonyms to the original query, thereby improving recall and addressing vocabulary mismatch. Query expansion techniques utilize methods such as synonym mapping, word embeddings, and co-occurrence analysis to broaden the scope of the search and retrieve more relevant information.",
    "An orange is a citrus fruit known for its round shape, bright orange color, and tangy flavor. Oranges are rich in vitamin C, fiber, and antioxidants, making them a popular choice for fresh consumption and juice extraction. They are cultivated in subtropical regions worldwide and are available in various varieties, including navel, Valencia, and blood oranges.",
    "Information extraction involves automatically extracting structured information from unstructured or semi-structured sources such as text documents, web pages, and databases. Information extraction systems use techniques such as pattern matching, natural language parsing, and named entity recognition to identify and extract relevant data elements.",
    "Named entity recognition and classification are fundamental tasks in natural language processing, with applications in information extraction, question answering, and text understanding. NER systems identify named entities in text and classify them into predefined categories, providing valuable information for downstream NLP tasks and applications.",
    "Automatic summarization systems can generate concise summaries of documents, saving time and effort for users who need to quickly grasp the main points of lengthy texts. Summarization systems employ techniques such as text clustering, sentence scoring, and content abstraction to produce coherent and informative summaries tailored to user preferences.",
]


tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(documents)


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

    return sorted(scores, key=lambda x: x["score"], reverse=True)


def semantic_result(query, top_k=2):
    query_embedding = embedding_model.encode([query])
    scores = cosine_similarity(query_embedding, document_embeddings)[0]
    top_doc_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[idx], scores[idx]) for idx in top_doc_indices]


query = "Natural Language processing for information retrieval"

retrieved_docs = semantic_result(query, 10)
for ri in retrieved_docs:
    print(ri[1], ri[0])
    print()

reranked_docs = colbert_rerank([ri[0] for ri in retrieved_docs], query)
print("Reranked documents:")
for ri in reranked_docs:
    print(ri["score"], ri["document"])
    print()
