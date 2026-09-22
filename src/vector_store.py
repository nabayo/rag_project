import uuid

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

COLLECTION_NAME = "documents_rag"
MODEL_EMBEDDING = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, chemin_persistance: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=chemin_persistance)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        self.model = SentenceTransformerEmbeddingFunction(
            model_name=MODEL_EMBEDDING,
            device="cpu",
            normalize_embeddings=False,
        )

    def add_chunks(self, chunks):
        """
        Ajoute des chunks à la collection.

        Parameters
        ----------
        chunks : list
            Liste contenant tous les chunks à ajouter à la collection.

        Returns
        ----------
        None
        """
        self.collection.add(
            documents=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[str(uuid.uuid4()) for _ in chunks],
        )

    def retrieval(self, query, top_k=5):
        """
        Recherche des chunks les plus proches sémantiquement.

        Parameters
        ----------
        query : str
            Requête de l'utilisateur.
        top_k : int, optional
            Nombre de chunks à retourner. Par défaut fixé à 5.

        Returns
        ----------
        res_list : list
            Liste de dictionnaires correspondant à chaque chunk retourné.
            Pour chaque chunk on retourne son contenu et les métadonnées associées.
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        res_list = []
        for doc, meta in zip(documents, metadatas):
            res_list.append({"content": doc, "metadata": meta})

        return res_list

    def retrieval_with_reranker(self, encoder, query, top_k=5):
        """
        Recherche des chunks les plus proches sémantiquement avec reranking.

        Parameters
        ----------
        encoder :
            Modèle d'encoder à utiliser.
        query : str
            Requête de l'utilisateur.
        top_k : int, optional
            Nombre de chunks à retourner. Par défaut fixé à 5.

        Returns
        ----------
        res_list : list
            Liste de dictionnaires correspondant à chaque chunk retourné.
            Pour chaque chunk on retourne son contenu, les métadonnées et le
            score associés.
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        pairs = [(query, doc) for doc in documents]

        scores = encoder.predict(pairs)
        combined = list(zip(scores, documents, metadatas))
        reranked = sorted(combined, key=lambda x: x[0], reverse=True)

        res_list = []
        for score, doc, meta in reranked:
            res_list.append(
                {"content": doc, "metadata": meta, "score": float(score)}
            )

        return res_list
