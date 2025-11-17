# utils/vectorstore.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from config.config import config

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss not installed or failed to import. Install faiss-cpu or faiss-gpu.") from e

# Files used for persistence
_INDEX_PATH = os.path.join(config.VECTORSTORE_DIR, "faiss.index")
_META_PATH = os.path.join(config.VECTORSTORE_DIR, "meta.pkl")
_IDMAP_PATH = os.path.join(config.VECTORSTORE_DIR, "id_map.pkl")  # maps index positions -> chunk_id

def _ensure_dir():
    os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)

def build_index_from_embeddings(
    embeddings: List[List[float]],
    chunk_ids: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    overwrite: bool = True
) -> None:
    """
    Build a new FAISS index from scratch using embeddings and save metadata.
    - embeddings: list of vectors (list of floats) shape (N, dim)
    - chunk_ids: list of string ids for each embedding (len N)
    - metadatas: optional list of metadata dicts parallel to embeddings
    """
    if not embeddings or not chunk_ids or len(embeddings) != len(chunk_ids):
        raise ValueError("Embeddings and chunk_ids must exist and have same length.")

    _ensure_dir()
    xb = np.array(embeddings).astype("float32")
    dim = xb.shape[1]

    # Use a simple flat L2 index (good for correctness & simplicity)
    index = faiss.IndexFlatL2(dim)
    index.add(xb)

    # persist index
    faiss.write_index(index, _INDEX_PATH)

    # persist id map (index position -> chunk id)
    with open(_IDMAP_PATH, "wb") as f:
        pickle.dump(chunk_ids, f)

    # persist metadata (chunk id -> metadata dict)
    if metadatas:
        id_to_meta = {cid: meta for cid, meta in zip(chunk_ids, metadatas)}
    else:
        id_to_meta = {cid: {} for cid in chunk_ids}
    with open(_META_PATH, "wb") as f:
        pickle.dump(id_to_meta, f)

def load_index_and_meta() -> Tuple[Optional[faiss.Index], Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Load the saved FAISS index, id_map (list of chunk_ids), and metadata dict (chunk_id -> meta).
    Returns (index, id_list, id_to_meta) or (None, None, None) if not found.
    """
    if not os.path.exists(_INDEX_PATH) or not os.path.exists(_IDMAP_PATH) or not os.path.exists(_META_PATH):
        return None, None, None
    index = faiss.read_index(_INDEX_PATH)
    with open(_IDMAP_PATH, "rb") as f:
        id_list = pickle.load(f)
    with open(_META_PATH, "rb") as f:
        id_to_meta = pickle.load(f)
    return index, id_list, id_to_meta

def query_index(
    query_embedding: List[float],
    top_k: int = 5
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Query the saved index with a single embedding.
    Returns list of tuples: (chunk_id, distance, metadata)
    Distance uses L2 (lower is closer).
    """
    index, id_list, id_to_meta = load_index_and_meta()
    if index is None:
        return []

    qv = np.array([query_embedding]).astype("float32")
    D, I = index.search(qv, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(id_list):
            continue
        cid = id_list[int(idx)]
        meta = id_to_meta.get(cid, {})
        results.append((cid, float(dist), meta))
    return results

def add_embeddings(
    embeddings: List[List[float]],
    chunk_ids: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Append new embeddings to an existing index.
    If index doesn't exist, it builds a new one.
    Note: FAISS IndexFlatL2 supports simple append via index.add.
    """
    if not embeddings or not chunk_ids or len(embeddings) != len(chunk_ids):
        raise ValueError("Embeddings and chunk_ids must exist and have same length.")

    _ensure_dir()
    xb = np.array(embeddings).astype("float32")

    # Load or create index
    index, id_list, id_to_meta = load_index_and_meta()
    if index is None:
        # build new index
        dim = xb.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(xb)
        id_list = list(chunk_ids)
        if metadatas:
            id_to_meta = {cid: meta for cid, meta in zip(chunk_ids, metadatas)}
        else:
            id_to_meta = {cid: {} for cid in chunk_ids}
    else:
        # ensure matching dims
        if xb.shape[1] != index.d:
            raise ValueError(f"Embedding dim mismatch. Index dim={index.d}, new embeddings dim={xb.shape[1]}")
        index.add(xb)
        # append ids & metadata
        id_list.extend(chunk_ids)
        if metadatas:
            for cid, meta in zip(chunk_ids, metadatas):
                id_to_meta[cid] = meta
        else:
            for cid in chunk_ids:
                id_to_meta[cid] = {}

    # persist everything
    faiss.write_index(index, _INDEX_PATH)
    with open(_IDMAP_PATH, "wb") as f:
        pickle.dump(id_list, f)
    with open(_META_PATH, "wb") as f:
        pickle.dump(id_to_meta, f)

def delete_vectorstore() -> None:
    """
    Utility to remove the stored index & metadata (useful during dev).
    """
    for p in (_INDEX_PATH, _IDMAP_PATH, _META_PATH):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

# ----------------- Quick test / usage example -------------------------------
if __name__ == "__main__":
    # Simple smoke test
    # Build small index, query, then append and query again
    emb1 = np.random.randn(10, 128).astype("float32").tolist()
    ids1 = [f"doc1-c{i}" for i in range(10)]
    metas1 = [{"source": "doc1", "page": 1, "chunk": i} for i in range(10)]
    delete_vectorstore()
    build_index_from_embeddings(emb1, ids1, metas1)
    q = emb1[3]  # query with an existing vector
    res = query_index(q, top_k=3)
    print("Initial query results:", res)

    # append more
    emb2 = np.random.randn(5, 128).astype("float32").tolist()
    ids2 = [f"doc2-c{i}" for i in range(5)]
    metas2 = [{"source": "doc2", "page": 2, "chunk": i} for i in range(5)]
    add_embeddings(emb2, ids2, metas2)
    res2 = query_index(emb2[2], top_k=4)
    print("After append query:", res2)
