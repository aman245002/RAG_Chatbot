# models/embeddings.py
import time
import os
from typing import List, Dict, Any, Tuple, Optional
from config.config import config

# Attempt imports lazily so module loads even if some deps aren't installed
_OPENAI_AVAILABLE = False
_SENTE_TRANS_AVAILABLE = False
try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTE_TRANS_AVAILABLE = True
except Exception:
    _SENTE_TRANS_AVAILABLE = False

# --- Utility: exponential backoff for API calls ----------------------------
def _backoff_sleep(attempt: int):
    # exponential backoff with jitter
    base = 0.8
    sleep = base * (2 ** attempt)
    jitter = min(1.0, sleep * 0.1)
    time.sleep(sleep + (jitter * (0.5 - os.urandom(1)[0] / 255.0)))  # tiny jitter

# --- OpenAI embedding helper -----------------------------------------------
class _OpenAIEmbedder:
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, max_retries: int = 5):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed. Install with `pip install openai`")
        self.model_name = model_name or config.OPENAI_EMBEDDING_MODEL or "text-embedding-3-small"
        self.max_retries = max_retries
        openai.api_key = api_key or config.OPENAI_API_KEY
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not configured (config.OPENAI_API_KEY is empty)")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embed list of texts using OpenAI embeddings API with batching and retry.
        Returns list of embeddings (as lists of floats) in the same order as input texts.
        """
        results = []
        n = len(texts)
        for i in range(0, n, batch_size):
            batch = texts[i:i+batch_size]
            attempt = 0
            while True:
                try:
                    resp = openai.Embedding.create(
                        model=self.model_name,
                        input=batch
                    )
                    # resp["data"] is aligned with batch
                    batch_embs = [item["embedding"] for item in resp["data"]]
                    results.extend(batch_embs)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise RuntimeError(f"OpenAI embedding failed after {self.max_retries} retries. Last error: {e}")
                    _backoff_sleep(attempt)
        return results

# --- Sentence-Transformers helper ------------------------------------------
class _LocalSTEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _SENTE_TRANS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed. Install with `pip install sentence-transformers`")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        # SentenceTransformer returns numpy arrays; convert to lists
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        return [e.tolist() for e in embs]

# --- Public EmbeddingEngine wrapper ----------------------------------------
class EmbeddingEngine:
    def __init__(self, prefer_openai: Optional[bool] = None):
        """
        prefer_openai: if True -> use OpenAI; if False -> use local sentence-transformers;
                       if None -> decide from config.USE_OPENAI_EMBEDDINGS.
        """
        if prefer_openai is None:
            prefer_openai = bool(config.USE_OPENAI_EMBEDDINGS)

        self._prefer_openai = prefer_openai
        self._engine = None

        if prefer_openai:
            # try OpenAI first
            if not _OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI library not available. Install `openai` or set USE_OPENAI_EMBEDDINGS=false")
            self._engine = _OpenAIEmbedder()
        else:
            # use sentence-transformers
            if not _SENTE_TRANS_AVAILABLE:
                raise RuntimeError("sentence-transformers not available. Install it or set USE_OPENAI_EMBEDDINGS=true")
            self._engine = _LocalSTEmbedder()

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Public method to embed list of strings. Returns list of list[float].
        """
        if not texts:
            return []
        if batch_size is None:
            batch_size = 64 if self._prefer_openai else 32
        return self._engine.embed_texts(texts, batch_size=batch_size)

    def embed_chunks(self, chunks: List[Dict[str, Any]], text_key: str = "text", batch_size: Optional[int] = None) -> Tuple[List[List[float]], List[str]]:
        """
        Given a list of chunk metadata dicts (as produced by doc_utils.create_chunks_from_file),
        extract texts using `text_key`, compute embeddings for each chunk, and return:
           (embeddings, list_of_chunk_ids)
        Embeddings are returned in the same order as the input chunks.
        """
        texts = [c.get(text_key, "") for c in chunks]
        ids = [c.get("id", str(i)) for i, c in enumerate(chunks)]
        embs = self.embed_texts(texts, batch_size=batch_size)
        # safety: ensure same length
        if len(embs) != len(ids):
            raise RuntimeError("Embedding count mismatch vs chunk ids")
        return embs, ids

# Convenience factory
def get_embedding_engine(prefer_openai: Optional[bool] = None) -> EmbeddingEngine:
    return EmbeddingEngine(prefer_openai=prefer_openai)

# --- Quick CLI test support -----------------------------------------------
if __name__ == "__main__":
    # quick smoke test (requires either OpenAI key or sentence-transformers)
    engine = None
    try:
        engine = get_embedding_engine(prefer_openai=config.USE_OPENAI_EMBEDDINGS)
    except Exception as e:
        print("Failed to initialize embedding engine:", e)
        raise SystemExit(1)

    sample_texts = [
        "This is a short test sentence.",
        "Another sentence about machine learning and embeddings."
    ]
    print("Embedding", len(sample_texts), "texts...")
    vectors = engine.embed_texts(sample_texts)
    print("Returned", len(vectors), "embeddings. Dim:", len(vectors[0]) if vectors else 0)
