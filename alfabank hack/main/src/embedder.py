# src/embedder.py
from typing import List
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from . import config

_E5_LIKE_PATTERNS = [r"\be5\b", r"\bgte\b", r"\bbge-m3\b", r"\b(text-)?embedding\b"]

def _is_e5_like(model_name: str) -> bool:
    name = model_name.lower()
    return any(re.search(p, name) for p in _E5_LIKE_PATTERNS)

class Embedder:
    """
    Обёртка над SentenceTransformer:
    - префиксы e5/gte/bge-m3 (query:/passage:)
    - L2-нормализация
    - батчи из config
    """
    def __init__(self, model_name: str = None, normalize: bool = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.normalize = config.EMBEDDING_NORMALIZE if normalize is None else normalize
        self._use_e5_prefix = _is_e5_like(self.model_name)

        self.model = SentenceTransformer(self.model_name, trust_remote_code=config.TRUST_REMOTE_CODE)

    def _maybe_prefix_queries(self, queries: List[str]) -> List[str]:
        if not self._use_e5_prefix:
            return queries
        return [f"query: {q}" for q in queries]

    def _maybe_prefix_passages(self, passages: List[str]) -> List[str]:
        if not self._use_e5_prefix:
            return passages
        return [f"passage: {p}" for p in passages]

    def encode_queries(self, queries: List[str], batch_size: int | None = None) -> np.ndarray:
        texts = self._maybe_prefix_queries(queries)
        bs = batch_size or config.QUERY_EMB_BATCH_SIZE
        emb = self.model.encode(
            texts, batch_size=bs, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=self.normalize
        )
        return emb

    def encode_passages(self, passages: List[str], batch_size: int | None = None) -> np.ndarray:
        texts = self._maybe_prefix_passages(passages)
        bs = batch_size or config.PASSAGE_EMB_BATCH_SIZE
        emb = self.model.encode(
            texts, batch_size=bs, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=self.normalize
        )
        return emb
