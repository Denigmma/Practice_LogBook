# src/preprocessor.py
from typing import Dict, List
import pandas as pd
import re
from dataclasses import dataclass
from io import StringIO
from transformers import AutoTokenizer
from . import config
import csv, sys

@dataclass
class Chunk:
    chunk_id: str
    web_id: int
    document_id: int
    url: str
    title: str
    text: str
    chunk_idx: int
    is_title_chunk: bool

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

def _clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _bump_field_size_limit():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

def _read_csv_robust(path):
    encodings = ["utf-8", "utf-8-sig", "cp1251", "windows-1251", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=",", engine="c", encoding=enc, quoting=csv.QUOTE_MINIMAL)
        except Exception as e:
            last_err = e
    _bump_field_size_limit()
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
        except Exception as e:
            last_err = e
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        return pd.read_csv(StringIO(txt), sep=",", engine="c")
    except Exception:
        pass
    try:
        with open(path, "r", encoding="latin1", errors="ignore") as f:
            txt = f.read()
        return pd.read_csv(StringIO(txt), sep=",", engine="c")
    except Exception:
        raise last_err if last_err is not None else RuntimeError(f"Failed to read CSV: {path}")

def _rename_like(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    low = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=low)

    def pick(cols, candidates):
        for cand in candidates:
            if cand in cols:
                return cand
        return None

    cols = set(df.columns)

    # websites: web_id/url/title/text
    if "web_id" not in cols:
        alt = pick(cols, ["id", "page_id", "doc_id"])
        if alt: df = df.rename(columns={alt: "web_id"})
    if "url" not in cols:
        alt = pick(cols, ["link", "href"])
        if alt: df = df.rename(columns={alt: "url"})
    if "title" not in cols:
        alt = pick(cols, ["page_title", "name", "header"])
        if alt: df = df.rename(columns={alt: "title"})
    if "text" not in cols:
        alt = pick(cols, ["content", "body", "parsed_text", "html_text"])
        if alt: df = df.rename(columns={alt: "text"})

    # questions: q_id/query
    if "q_id" not in cols:
        alt = pick(cols, ["id", "question_id"])
        if alt: df = df.rename(columns={alt: "q_id"})
    if "query" not in cols:
        alt = pick(cols, ["question", "q_text", "text"])
        if alt: df = df.rename(columns={alt: "query"})

    return df

# ----------------- Query normalization -----------------
_REPLACERS = [
    (r"\bр/?с\b", "расчетный счет"),
    (r"\bбик\b", "бик"),
    (r"\bинн\b", "инн"),
    (r"\bкпп\b", "кпп"),
    (r"\bк/с\b", "корр счет"),
]

def normalize_query(q: str) -> str:
    q = _clean_text(q.lower())
    for pat, rep in _REPLACERS:
        q = re.sub(pat, rep, q)
    return q

# ----------------- Chunking -----------------
class TokenChunker:
    """Chunk text by model tokenizer tokens."""
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=config.TRUST_REMOTE_CODE
        )
        self.chunk_tokens = config.CHUNK_TOKENS
        self.overlap_tokens = config.OVERLAP_TOKENS

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        step = self.chunk_tokens - self.overlap_tokens
        for start in range(0, len(tokens), step):
            piece = tokens[start : start + self.chunk_tokens]
            if not piece:
                break
            chunks.append(self.tokenizer.decode(piece, skip_special_tokens=True))
            if start + self.chunk_tokens >= len(tokens):
                break
        return chunks

def create_chunks_from_websites(df_web: pd.DataFrame, model_name: str) -> List[Chunk]:
    tk = TokenChunker(model_name)
    chunks: List[Chunk] = []
    for row in df_web.itertuples(index=False):
        web_id = int(getattr(row, "web_id"))
        url = _clean_text(getattr(row, "url", ""))
        title = _clean_text(getattr(row, "title", ""))
        body = _clean_text(getattr(row, "text", ""))

        lead = body[:600] if body else ""
        title_payload = (title + " " + lead).strip() if title or lead else ""
        idx = 0
        if title_payload:
            chunks.append(Chunk(
                chunk_id=f"{web_id}::t0", web_id=web_id, document_id=web_id,
                url=url, title=title, text=title_payload, chunk_idx=idx, is_title_chunk=True
            ))
            idx += 1
        for piece in tk.chunk_text(body):
            chunks.append(Chunk(
                chunk_id=f"{web_id}::{idx}", web_id=web_id, document_id=web_id,
                url=url, title=title, text=piece, chunk_idx=idx, is_title_chunk=False
            ))
            idx += 1
    return chunks

def load_corpus() -> pd.DataFrame:
    df = _read_csv_robust(config.WEBSITES_CSV)
    df = _rename_like(df, {})
    need = ["web_id", "url", "title", "text"]
    for col in need:
        if col not in df.columns:
            if col in ("title", "text"):
                df[col] = ""
            else:
                raise ValueError(f"В websites_updated.csv не найдена колонка '{col}'.")
        else:
            df[col] = df[col].fillna("")
    return df[need]

def load_queries() -> pd.DataFrame:
    dfq = _read_csv_robust(config.QUESTIONS_CSV)
    dfq = _rename_like(dfq, {})
    if "q_id" not in dfq.columns or "query" not in dfq.columns:
        raise ValueError("В questions_clean.csv не найдены колонки q_id и/или query (или их синонимы).")
    dfq["query"] = dfq["query"].fillna("").map(normalize_query)
    return dfq[["q_id", "query"]]
