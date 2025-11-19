# src/reranker.py
from typing import List, Optional
import torch
from sentence_transformers import CrossEncoder
from . import config
import logging

logger = logging.getLogger("reranker")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

_FALLBACKS = [
    "BAAI/bge-reranker-v2-m3",                 # мультиязычный SOTA
    "cross-encoder/ms-marco-MiniLM-L-6-v2",    # быстрый англ. фолбэк
]

class CrossEncoderReranker:
    """CrossEncoder (query, doc_text) с фолбэками и выбором устройства."""
    def __init__(self, prefer: Optional[str] = "BAAI/bge-reranker-v2-m3"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        errors = []
        tried = [prefer] + _FALLBACKS if prefer else _FALLBACKS
        self.model = None
        for repo in tried:
            if not repo:
                continue
            try:
                self.model = CrossEncoder(
                    repo,
                    device=device,
                    max_length=config.RERANKER_MAX_LENGTH
                )
                print(f"[RERANKER] loaded: {repo} on {device}")
                break
            except Exception as e:
                errors.append(f"{repo}: {e}")
                continue
        if self.model is None:
            raise RuntimeError("Failed to load any CrossEncoder reranker.\n" + "\n".join(errors))

    def score(self, query: str, docs: List[str], batch_size: int | None = None) -> List[float]:
        if not docs:
            return []
        bs = batch_size or config.RERANKER_BATCH_SIZE
        pairs = [[query, d] for d in docs]
        # Включаем прогресс-бар библиотеки, чтобы видеть ход обработки кандидатов
        with torch.inference_mode():
            scores = self.model.predict(
                pairs,
                batch_size=bs,
                convert_to_numpy=True,
                show_progress_bar=True  # <— ключевой индикатор прогресса
            )
        return [float(s) for s in scores]
