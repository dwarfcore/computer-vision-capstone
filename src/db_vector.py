"""
db_vector.py - Vector database layer for face embedding storage and similarity search.

Local:  ChromaDB (file-backed, no server needed)
Remote: Qdrant (self-hosted) via HTTP REST API
Cloud:  Pinecone (managed) via pinecone-client

Each backend implements the same interface:
    VectorDB.add(person_id, embedding)
    VectorDB.search(embedding, top_k) -> [(person_id, score), ...]
    VectorDB.delete(person_id)
    VectorDB.count() -> int
"""
import os
import numpy as np
from typing import List, Tuple, Optional
from config import AppConfig, DatabaseMode

COLLECTION_NAME = "face_embeddings"


# ─────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────

class VectorDB:
    def add(self, person_id: int, embedding: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Returns list of (person_id, distance) sorted by distance ascending."""
        raise NotImplementedError

    def delete(self, person_id: int) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ─────────────────────────────────────────────
#  ChromaDB (local)
# ─────────────────────────────────────────────

class ChromaVectorDB(VectorDB):
    def __init__(self, persist_dir: str):
        import chromadb
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, person_id: int, embedding: np.ndarray) -> None:
        emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        uid = f"person_{person_id}"
        # Upsert: delete existing then re-add
        try:
            self._col.delete(ids=[uid])
        except Exception:
            pass
        self._col.add(
            ids=[uid],
            embeddings=[emb_list],
            metadatas=[{"person_id": person_id}]
        )

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        results = self._col.query(
            query_embeddings=[emb_list],
            n_results=min(top_k, max(self._col.count(), 1))
        )
        hits = []
        if results and results.get("ids") and results["ids"][0]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                hits.append((meta["person_id"], dist))
        return hits

    def delete(self, person_id: int) -> None:
        try:
            self._col.delete(ids=[f"person_{person_id}"])
        except Exception:
            pass

    def count(self) -> int:
        return self._col.count()


# ─────────────────────────────────────────────
#  Qdrant (remote self-hosted)
# ─────────────────────────────────────────────

class QdrantVectorDB(VectorDB):
    """
    Requires: pip install qdrant-client
    Run Qdrant locally: docker run -p 6333:6333 qdrant/qdrant
    """
    def __init__(self, url: str, vector_size: int = 128):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._client = QdrantClient(url=url)
        self._size = vector_size
        self._collection = COLLECTION_NAME
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def add(self, person_id: int, embedding: np.ndarray) -> None:
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=person_id, vector=embedding.tolist(), payload={"person_id": person_id})]
        )

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        results = self._client.search(
            collection_name=self._collection,
            query_vector=embedding.tolist(),
            limit=top_k
        )
        return [(r.payload["person_id"], 1.0 - r.score) for r in results]

    def delete(self, person_id: int) -> None:
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[person_id])
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.vectors_count or 0


# ─────────────────────────────────────────────
#  Pinecone (cloud managed)
# ─────────────────────────────────────────────

class PineconeVectorDB(VectorDB):
    """
    Requires: pip install pinecone-client
    Set cloud_vector_url to your Pinecone index host,
    cloud_vector_api_key to your API key.
    """
    def __init__(self, api_key: str, index_host: str, vector_size: int = 128):
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(host=index_host)

    def add(self, person_id: int, embedding: np.ndarray) -> None:
        self._index.upsert(vectors=[{
            "id": f"person_{person_id}",
            "values": embedding.tolist(),
            "metadata": {"person_id": person_id}
        }])

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        res = self._index.query(vector=embedding.tolist(), top_k=top_k, include_metadata=True)
        return [(int(m["metadata"]["person_id"]), 1.0 - m["score"]) for m in res["matches"]]

    def delete(self, person_id: int) -> None:
        self._index.delete(ids=[f"person_{person_id}"])

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.get("total_vector_count", 0)


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

def create_vector_db(cfg: AppConfig) -> VectorDB:
    mode = cfg.db.mode
    if mode == DatabaseMode.LOCAL:
        return ChromaVectorDB(persist_dir=cfg.db.local_vector_path)
    elif mode == DatabaseMode.REMOTE:
        url = cfg.db.remote_vector_url or "http://localhost:6333"
        return QdrantVectorDB(url=url)
    elif mode == DatabaseMode.CLOUD:
        if not cfg.db.cloud_vector_api_key:
            print("[VectorDB] Cloud API key not set — falling back to local ChromaDB")
            return ChromaVectorDB(persist_dir=cfg.db.local_vector_path)
        return PineconeVectorDB(
            api_key=cfg.db.cloud_vector_api_key,
            index_host=cfg.db.cloud_vector_url
        )
    raise ValueError(f"Unknown mode: {mode}")
