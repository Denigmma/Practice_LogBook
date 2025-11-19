# src/retriever.py
from typing import List, Dict, Tuple, Optional
import math, time, re
from collections import defaultdict
import logging
import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
import torch

from . import config
from .embedder import Embedder
from .preprocessor import create_chunks_from_websites, load_corpus, Chunk
from .bm25 import BM25ChunkIndex

try:
    from .reranker import CrossEncoderReranker  # noqa: F401
except Exception:
    CrossEncoderReranker = None  # type: ignore

# ---- logging ----
logger = logging.getLogger("retriever")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
def re_split_simple(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")

_POS_PATTS = [re.compile(p, re.I) for p in config.URL_POSITIVE_PATTERNS]
_NEG_PATTS = [re.compile(p, re.I) for p in config.URL_NEGATIVE_PATTERNS]

def _pos_url_title_bonus(url: str, title: str) -> float:
    s = url + " " + (title or "")
    return config.URL_POSITIVE_BOOST if any(p.search(s) for p in _POS_PATTS) else 0.0

def _neg_url_penalty(url: str) -> float:
    return config.URL_NEGATIVE_PENALTY if any(p.search(url) for p in _NEG_PATTS) else 0.0

def _min_max_norm(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax):
        return {k: 0.0 for k in scores}
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}

def _is_short_query(q: str) -> bool:
    toks = [t for t in re_split_simple(q) if len(t) > 2]
    return len(toks) <= config.SHORT_QUERY_TOKENS

def _has_digits(q: str) -> bool:
    return any(ch.isdigit() for ch in q)

def _has_keywords(q: str) -> bool:
    low = q.lower()
    return any(kw in low for kw in config.BM25_BONUS_KEYWORDS)

def _dynamic_weights(q: str) -> Tuple[float, float]:
    d, b = config.WEIGHT_DENSE, config.WEIGHT_BM25
    if not config.USE_DYNAMIC_WEIGHTS:
        return d, b
    bonus = 0.0
    if _is_short_query(q): bonus += config.BM25_BONUS_SHORT
    if _has_digits(q):     bonus += config.BM25_BONUS_HAS_DIGITS
    if _has_keywords(q):   bonus += 0.15
    b_new = min(0.9, max(0.1, b + bonus))
    d_new = 1.0 - b_new
    return d_new, b_new

def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]

def _sent_score(sent: str, q_tokens: set[str]) -> float:
    toks = {t.lower() for t in re_split_simple(sent) if len(t) > 2}
    return float(len(toks & q_tokens))

class AlfabankRetrieval:
    """Dense (Chroma) + BM25 гибрид + CrossEncoder reranker (с прогрессом)."""
    def __init__(self):
        self.embedder = Embedder(model_name=config.EMBEDDING_MODEL, normalize=config.EMBEDDING_NORMALIZE)
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._chunk_meta: Dict[str, Chunk] = {}  # chunk_id -> Chunk
        self._doc_info: Dict[int, Tuple[str, str]] = {}  # doc_id -> (url, title)
        self._bm25: Optional[BM25ChunkIndex] = None

        self._reranker = (CrossEncoderReranker()  # type: ignore
                          if (getattr(config, "USE_RERANKER", False) and CrossEncoderReranker is not None)
                          else None)

    # ---------- Build index ----------
    def build_index(self) -> None:
        logger.info("Loading corpus...")
        df_web = load_corpus()
        logger.info(f"Corpus: {len(df_web)} pages")

        logger.info("Chunking pages (token-based)...")
        chunks = create_chunks_from_websites(df_web, model_name=config.EMBEDDING_MODEL)
        logger.info(f"Total chunks: {len(chunks)}")

        self._chunk_meta = {c.chunk_id: c for c in chunks}
        for c in chunks:
            if c.document_id not in self._doc_info:
                self._doc_info[c.document_id] = (c.url, c.title)

        if chunks:
            try:
                self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
                logger.info("Old Chroma collection removed.")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )

        logger.info("Adding chunks to Chroma with precomputed embeddings...")
        ids_all = [c.chunk_id for c in chunks]
        metas_all = [{
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "web_id": c.web_id,
            "url": c.url,
            "title": c.title,
            "is_title_chunk": c.is_title_chunk,
            "chunk_idx": c.chunk_idx,
        } for c in chunks]
        docs_all = [c.text for c in chunks]

        t0 = time.time()
        for start in range(0, len(ids_all), config.BATCH_ADD_SIZE):
            end = min(start + config.BATCH_ADD_SIZE, len(ids_all))
            docs_batch = docs_all[start:end]
            ids_batch = ids_all[start:end]
            metas_batch = metas_all[start:end]
            emb_batch = self.embedder.encode_passages(docs_batch)
            self.collection.add(
                ids=ids_batch,
                documents=docs_batch,
                metadatas=metas_batch,
                embeddings=emb_batch.tolist(),
            )
            logger.info(f"Added chunks {start}..{end-1} (elapsed {time.time()-t0:.1f}s)")

        if config.USE_BM25:
            logger.info("Building BM25 index over chunks...")
            self._bm25 = BM25ChunkIndex.build_from_chunks(chunks)
            self._bm25.save(str(config.BM25_INDEX_PATH))
            logger.info(f"BM25 saved to {config.BM25_INDEX_PATH}")

        logger.info("Index build completed.")

    def _ensure_bm25_loaded(self) -> None:
        if not config.USE_BM25: return
        if self._bm25 is not None: return
        try:
            self._bm25 = BM25ChunkIndex.load(str(config.BM25_INDEX_PATH))
            logger.info("BM25 index loaded from disk.")
        except Exception:
            logger.warning("BM25 index not found. Run build_index() first.")
            self._bm25 = None

    # ---------- Dense candidates (batched) ----------
    def _dense_candidates_batch(
        self, q_vecs: np.ndarray, n_results: int
    ) -> Tuple[List[List[Tuple[str, float]]], List[Dict[str, int]]]:
        res = self.collection.query(
            query_embeddings=[q.tolist() for q in q_vecs],
            n_results=n_results,
            include=["metadatas", "distances"]
        )
        ids_list = res.get("ids", [])
        dists_list = res.get("distances", [])
        metas_list = res.get("metadatas", [])

        out_pairs: List[List[Tuple[str, float]]] = []
        out_maps: List[Dict[str, int]] = []
        for ids, dists, metas in zip(ids_list, dists_list, metas_list):
            if not ids:
                out_pairs.append([]); out_maps.append({}); continue
            sims = [1.0 - float(d) for d in dists]
            pairs = list(zip(ids, sims))
            pairs.sort(key=lambda x: x[1], reverse=True)
            id2doc: Dict[str, int] = {}
            for cid, md in zip(ids, metas):
                try:
                    id2doc[cid] = int(md.get("document_id"))
                except Exception:
                    pass
            out_pairs.append(pairs)
            out_maps.append(id2doc)
        return out_pairs, out_maps

    # ---------- pooling ----------
    def _topk_pool_per_doc(
        self,
        pairs: List[Tuple[str, float]],
        k: int,
        id2doc_hint: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[int, float], Dict[int, List[Tuple[str, float]]]]:
        per_doc: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        for chunk_id, score in pairs:
            doc_id = None
            if id2doc_hint and chunk_id in id2doc_hint:
                doc_id = id2doc_hint[chunk_id]
            else:
                meta = self._chunk_meta.get(chunk_id)
                if meta:
                    doc_id = meta.document_id
            if doc_id is None:
                continue
            per_doc[doc_id].append((chunk_id, float(score)))

        doc_scores: Dict[int, float] = {}
        doc_chunks: Dict[int, List[Tuple[str, float]]] = {}
        for doc_id, vals in per_doc.items():
            vals.sort(key=lambda x: x[1], reverse=True)
            pool = vals[:k]
            if pool:
                smax = pool[0][1]
                mean_k = float(np.mean([s for _, s in pool]))
                doc_scores[doc_id] = float(smax + 0.5 * mean_k)
                doc_chunks[doc_id] = pool
            else:
                doc_scores[doc_id] = 0.0
                doc_chunks[doc_id] = []
        return doc_scores, doc_chunks

    # ---------- Main batched search ----------
    def search(self, query: str, top_k: Optional[int] = None) -> List[int]:
        return self.search_batch([query], top_k=top_k or config.FINAL_TOP_K_DOCS)[0]

    def search_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[List[int]]:
        if not queries:
            return []
        top_k = top_k or config.FINAL_TOP_K_DOCS

        t_start = time.time()
        logger.info(f"[search_batch] start: {len(queries)} queries")

        # 1) batch-encode queries
        t0 = time.time()
        q_vecs = self.embedder.encode_queries(queries, batch_size=config.QUERY_EMB_BATCH_SIZE)
        if not isinstance(q_vecs, np.ndarray):
            q_vecs = np.asarray(q_vecs, dtype=np.float32)
        logger.info(f"[search_batch] embed {len(queries)} q in {time.time()-t0:.2f}s")

        # 2) batched dense retrieval
        t1 = time.time()
        dense_pairs_list, id2doc_list = self._dense_candidates_batch(
            q_vecs, n_results=config.SEARCH_TOP_K_CHUNKS_DENSE
        )
        logger.info(f"[search_batch] dense candidates in {time.time()-t1:.2f}s")

        results: List[List[int]] = []
        processed = 0
        every = max(1, getattr(config, "SEARCH_LOG_EVERY", 8))

        # Настройка троттлинга rerank на CPU
        local_rerank_k = config.RERANK_CANDIDATE_DOCS
        local_bs = config.RERANKER_BATCH_SIZE
        if getattr(config, "THROTTLE_RERANK_ON_CPU", True) and not torch.cuda.is_available():
            local_rerank_k = min(local_rerank_k, config.RERANK_CANDIDATE_DOCS_CPU)
            local_bs = max(1, config.RERANKER_BATCH_SIZE_CPU)
            logger.info(f"[search_batch] CPU detected → rerank candidates: {local_rerank_k}, batch_size: {local_bs}")

        for idx, (q_text, pairs, id2doc) in enumerate(zip(queries, dense_pairs_list, id2doc_list), 1):
            t_q = time.time()

            # dynamic weights
            w_dense, w_bm25 = _dynamic_weights(q_text)

            # dense pooling
            dense_doc, dense_chunks = self._topk_pool_per_doc(pairs, k=config.AGG_TOP_K_PER_DOC, id2doc_hint=id2doc)

            # bm25
            bm25_doc: Dict[int, float] = {}
            bm25_chunks: Dict[int, List[Tuple[str, float]]] = {}
            if config.USE_BM25:
                self._ensure_bm25_loaded()
                add = 150 if _is_short_query(q_text) or _has_keywords(q_text) else 0
                k_bm = config.SEARCH_TOP_K_CHUNKS_BM25 + add
                bm25_pairs = self._bm25.query(q_text, top_k=k_bm) if self._bm25 else []
                bm25_doc, bm25_chunks = self._topk_pool_per_doc(bm25_pairs, k=config.AGG_TOP_K_PER_DOC)

            # fusion
            dense_norm = _min_max_norm(dense_doc)
            bm25_norm = _min_max_norm(bm25_doc) if config.USE_BM25 else {}
            all_docs = set(dense_norm.keys()) | set(bm25_norm.keys())
            hybrid_scores: Dict[int, float] = {d: w_dense * dense_norm.get(d, 0.0) + w_bm25 * bm25_norm.get(d, 0.0)
                                               for d in all_docs}

            # title boost
            if config.TITLE_BOOST > 0:
                q_tokens = {t.lower() for t in re_split_simple(q_text) if len(t) > 2}
                for doc_id, (_, title) in self._doc_info.items():
                    title_tokens = {t.lower() for t in re_split_simple(title) if len(t) > 2}
                    if q_tokens & title_tokens and doc_id in hybrid_scores:
                        hybrid_scores[doc_id] *= (1.0 + config.TITLE_BOOST)

            # heuristics
            def apply_domain(hs: Dict[int, float]) -> Dict[int, float]:
                out = {}
                q_low = q_text.lower()
                is_premium_intent = any(x in q_low for x in ["a-club", "аклуб", "премиум", "private", "wealth"])
                for doc_id, s in hs.items():
                    url, title = self._doc_info.get(doc_id, ("", ""))
                    bonus = _pos_url_title_bonus(url, title)
                    malus = 0.0 if is_premium_intent else _neg_url_penalty(url)
                    out[doc_id] = s * (1.0 + bonus) * (1.0 - malus)
                return out

            hybrid_scores = apply_domain(hybrid_scores)
            prelim_ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            prelim_docs = [doc for doc, _ in prelim_ranked]

            # Build CE texts (лучшие предложения из топ-чанков)
            def build_doc_text(doc_id: int) -> str:
                title = self._doc_info.get(doc_id, ("", ""))[1] or ""
                merged: List[Tuple[str, float]] = []
                for dct in (dense_chunks, bm25_chunks):
                    if doc_id in dct:
                        merged.extend(dct[doc_id])
                seen = set()
                merged = sorted(merged, key=lambda x: x[1], reverse=True)
                uniq: List[Tuple[str, float]] = []
                for cid, s in merged:
                    if cid not in seen:
                        uniq.append((cid, s)); seen.add(cid)

                parts: List[str] = []
                q_tokens = {t.lower() for t in re_split_simple(q_text) if len(t) > 2}
                for cid, _ in uniq[:config.RERANKER_CHUNKS_PER_DOC]:
                    ch = self._chunk_meta.get(cid)
                    if not ch: continue
                    sents = _sentence_split(ch.text)
                    scored = sorted(sents, key=lambda st: _sent_score(st, q_tokens), reverse=True)
                    take = max(1, min(3, len(scored)))
                    parts.extend(scored[:take])

                text_blob = (title + "\n\n" + "\n".join(parts)).strip()
                if len(text_blob) > config.RERANKER_MAX_CHARS_PER_DOC:
                    text_blob = text_blob[:config.RERANKER_MAX_CHARS_PER_DOC]
                return text_blob

            # Rerank (с прогресс-баром внутри .score)
            if getattr(config, "USE_RERANKER", False) and (self._reranker is not None):
                cand_docs = prelim_docs[:local_rerank_k]
                cand_texts = [build_doc_text(d) for d in cand_docs]
                ce_scores = self._reranker.score(q_text, cand_texts, batch_size=local_bs)
                if ce_scores:
                    ce_min, ce_max = float(min(ce_scores)), float(max(ce_scores))
                    ce_norm = [0.0 if math.isclose(ce_min, ce_max) else (s - ce_min) / (ce_max - ce_min) for s in ce_scores]
                    mix_scores: Dict[int, float] = {}
                    for d, base_s, ce_s in zip(cand_docs, [hybrid_scores.get(d, 0.0) for d in cand_docs], ce_norm):
                        mix_scores[d] = (1.0 - config.RERANKER_ALPHA) * base_s + config.RERANKER_ALPHA * ce_s
                    final_ranked = sorted(mix_scores.items(), key=lambda x: x[1], reverse=True)
                    top_docs = [doc for doc, _ in final_ranked[:top_k]]
                else:
                    top_docs = prelim_docs[:top_k]
            else:
                top_docs = prelim_docs[:top_k]

            # добор до top_k
            if len(top_docs) < top_k:
                seen = set(top_docs)
                for d in prelim_docs:
                    if d not in seen:
                        top_docs.append(d); seen.add(d)
                    if len(top_docs) == top_k:
                        break

            uniq_docs: List[int] = []
            for d in top_docs:
                if d not in uniq_docs:
                    uniq_docs.append(d)
                if len(uniq_docs) == top_k:
                    break

            results.append(uniq_docs)
            processed += 1

            # прогресс внутри чанка
            if processed == 1 or processed % every == 0 or processed == len(queries):
                elapsed = time.time() - t_start
                rate = processed / max(elapsed, 1e-6)
                eta = (len(queries) - processed) / max(rate, 1e-6)
                logger.info(f"[search_batch] {processed}/{len(queries)} | last {time.time()-t_q:.2f}s | "
                            f"avg {elapsed/processed:.2f}s/q | ETA {eta/60:.1f} min")

        return results

    # ---------- batch_answer ----------
    def batch_answer(self, dfq: pd.DataFrame) -> pd.DataFrame:
        n = len(dfq)
        if n == 0:
            return pd.DataFrame(columns=["q_id", "web_list"])
        logger.info(f"[batch_answer] Start answering {n} queries...")
        t0 = time.time()
        rows: List[Dict[str, object]] = []
        chunk = max(1, getattr(config, "BATCH_ANSWER_CHUNK", 64))
        for i_start in range(0, n, chunk):
            i_end = min(i_start + chunk, n)
            df_chunk = dfq.iloc[i_start:i_end]
            queries = df_chunk["query"].tolist()
            q_ids = df_chunk["q_id"].astype(int).tolist()
            t_chunk = time.time()
            docs_lists = self.search_batch(queries, top_k=config.FINAL_TOP_K_DOCS)
            dur = time.time() - t_chunk
            for q_id, docs in zip(q_ids, docs_lists):
                while len(docs) < config.FINAL_TOP_K_DOCS:
                    docs.append(docs[-1] if docs else 1)
                rows.append({"q_id": q_id, "web_list": docs})
            done = i_end
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            logger.info(f"[batch_answer] {done}/{n} done | chunk {i_end-i_start} in {dur:.2f}s | "
                        f"avg {elapsed/done:.2f}s/q | {rate:.2f} q/s | ETA {eta/60:.1f} min")
        logger.info(f"[batch_answer] Finished {n} queries in {time.time()-t0:.1f}s")
        return pd.DataFrame(rows)

    # ---------- shutdown ----------
    def close(self) -> None:
        try:
            vs = getattr(self, "collection", None)
            if vs is not None:
                persist = getattr(vs, "persist", None)
                if callable(persist):
                    persist()
        except Exception:
            pass
        try:
            client = getattr(self, "client", None)
            for m in ("close", "reset", "shutdown", "teardown"):
                fn = getattr(client, m, None)
                if callable(fn):
                    try: fn()
                    except Exception: pass
        except Exception:
            pass
        try:
            self._bm25 = None
            self._chunk_meta.clear()
            self._doc_info.clear()
        except Exception:
            pass
