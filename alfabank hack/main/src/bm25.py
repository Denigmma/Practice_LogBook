# src/bm25.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
import re
import pickle

from rank_bm25 import BM25Okapi
from . import config

# Конфиги с дефолтами
_BM25_STEMMING = getattr(config, "BM25_STEMMING", False)
_BM25_NORMALIZE_E = getattr(config, "BM25_NORMALIZE_E", True)
_BM25_MIN_TOKEN_LEN = getattr(config, "BM25_MIN_TOKEN_LEN", 2)
_BM25_SYNONYMS: Dict[str, List[str]] = getattr(config, "BM25_SYNONYMS", {})

# Опциональный стеммер
try:
    import snowballstemmer  # type: ignore
    _STEMMER = snowballstemmer.stemmer("russian") if _BM25_STEMMING else None
except Exception:
    _STEMMER = None

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

def _normalize_token(t: str) -> str:
    t = t.lower()
    if _BM25_NORMALIZE_E:
        t = t.replace("ё", "е")
    return t

def _stem(t: str) -> str:
    if not _STEMMER:
        return t
    return _STEMMER.stemWord(t)

def _tokenize_ru(text: str) -> List[str]:
    if not text:
        return []
    toks = _WORD_RE.findall(text)
    out = []
    for t in toks:
        tt = _normalize_token(t)
        if len(tt) < _BM25_MIN_TOKEN_LEN:
            continue
        out.append(_stem(tt))
    return out

def _expand_query_tokens(tokens: List[str]) -> List[str]:
    """Примитивная синонимизация запроса под домен банка."""
    if not _BM25_SYNONYMS:
        return tokens
    bag = set(tokens)
    for canon, variants in _BM25_SYNONYMS.items():
        all_forms = [canon] + list(variants)
        norm_forms = [_stem(_normalize_token(x)) for x in all_forms]
        if any(f in bag for f in norm_forms):
            bag.update(norm_forms)
    return list(bag)

@dataclass
class _ChunkLite:
    chunk_id: str
    text: str

class BM25ChunkIndex:
    def __init__(self, tokenized_corpus: List[List[str]], chunks: List[_ChunkLite]):
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunk_ids = [c.chunk_id for c in chunks]

    @classmethod
    def build_from_chunks(cls, chunks_full: Iterable) -> "BM25ChunkIndex":
        chunks = [_ChunkLite(c.chunk_id, c.text) for c in chunks_full]
        tokenized = [_tokenize_ru(c.text) for c in chunks]
        return cls(tokenized, chunks)

    def query(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        q_tokens = _tokenize_ru(query)
        q_tokens = _expand_query_tokens(q_tokens)
        scores = self._bm25.get_scores(q_tokens)  # np.ndarray
        order = scores.argsort()[::-1][:top_k]
        out: List[Tuple[str, float]] = []
        for idx in order:
            out.append((self._chunk_ids[int(idx)], float(scores[int(idx)])))
        return out

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunk_ids": self._chunk_ids,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BM25ChunkIndex":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        inst = object.__new__(cls)
        inst._bm25 = obj["bm25"]
        inst._chunk_ids = obj["chunk_ids"]
        return inst
