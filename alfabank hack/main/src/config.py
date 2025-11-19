# src/config.py
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

WEBSITES_CSV = DATA_DIR / "websites_updated.csv"
QUESTIONS_CSV = DATA_DIR / "questions_clean.csv"
CHROMA_COLLECTION_NAME = "alfabank_chunks"

LOG_LEVEL = "INFO"

# ---------- Embeddings ----------
# Сильная мультиязычная модель (замена gte-base)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_NORMALIZE = True
TRUST_REMOTE_CODE = True

# Батчи энкодинга (ускоряют инференс)
QUERY_EMB_BATCH_SIZE = 128
PASSAGE_EMB_BATCH_SIZE = 128

# ---------- Chunking ----------
CHUNK_TOKENS = 256
OVERLAP_TOKENS = 64
BATCH_ADD_SIZE = 256

# ---------- Hybrid retrieval ----------
USE_BM25 = True
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"

# Больше кандидатов для более «жадного» поиска
SEARCH_TOP_K_CHUNKS_DENSE = 200
SEARCH_TOP_K_CHUNKS_BM25 = 300
AGG_TOP_K_PER_DOC = 4

# Базовые веса (переопределяются динамикой)
WEIGHT_DENSE = 0.60
WEIGHT_BM25 = 0.40
FINAL_TOP_K_DOCS = 5
TITLE_BOOST = 0.15

# Динамика фьюжна по типу запроса
USE_DYNAMIC_WEIGHTS = True
SHORT_QUERY_TOKENS = 3      # если <=3 токенов — считаем «коротким»
BM25_BONUS_SHORT = 0.20     # +20% к доле BM25 на коротких запросах
BM25_BONUS_HAS_DIGITS = 0.20  # +20% если есть цифры (реквизиты/номера)
BM25_BONUS_KEYWORDS = ["бик", "р/с", "рс", "инн", "кпп", "счет", "счёт", "номер"]

# URL/title эвристики
URL_POSITIVE_PATTERNS = [r"a-?club", r"private", r"wealth", r"invest", r"premium", r"rekvizit|реквизит"]
URL_NEGATIVE_PATTERNS = [r"vacan", r"press", r"news", r"cookies", r"legal", r"disclaimer"]
URL_POSITIVE_BOOST = 0.08
URL_NEGATIVE_PENALTY = 0.10

# ---------- Cross-Encoder reranker ----------
USE_RERANKER = True  # ВКЛЮЧАЕМ — это основной буст качества

# Сколько документов мы максимум отдаём на rerank (общий лимит)
# (на CPU будет дополнительно сжато до RERANK_CANDIDATE_DOCS_CPU)
RERANK_CANDIDATE_DOCS = 80  # было 120

RERANKER_ALPHA = 0.45

# Дефолтный batch_size для CrossEncoder (например, если будет GPU)
RERANKER_BATCH_SIZE = 32     # было 16

# Ограничения по размеру текста для CrossEncoder
RERANKER_MAX_LENGTH = 512
RERANKER_CHUNKS_PER_DOC = 5
RERANKER_MAX_CHARS_PER_DOC = 2000

# ---------- Batch answering ----------
BATCH_ANSWER_CHUNK = 64

# ---------- BM25 ----------
BM25_STEMMING = True
BM25_NORMALIZE_E = True
BM25_MIN_TOKEN_LEN = 2
BM25_SYNONYMS = {
    "кредит": ["займ", "рассрочка", "ипотека", "кредитная карта", "кредитка"],
    "счет": ["счёт", "account", "р/с", "рс", "расчетный счет", "зарплатный"],
    "перевод": ["трансфер", "платеж", "платёж", "оплата", "перечисление"],
    "брокерский": ["инвестиции", "инвест", "акции", "облигации"],
    "блокировка": ["заблокирован", "заблокировали", "бан", "блок"],
    "реквизиты": ["бик", "инн", "кпп", "коррсчет", "корр.счет", "корр счёт"],
}

# ---------- Progress / Throttle ----------
# как часто печатать прогресс внутри search_batch (в запросах)
SEARCH_LOG_EVERY = 8
# как часто печатать прогресс внутри reranker (включен прогресс-бар)
RERANK_LOG_EVERY = 8  # (пока не используется напрямую — прогресс-бар покажет ход)

# если нет GPU, автоматически уменьшаем объём rerank'а
THROTTLE_RERANK_ON_CPU = True

# На CPU берём меньше кандидатов, но увеличиваем batch_size для CrossEncoder
RERANK_CANDIDATE_DOCS_CPU = 32       # было 60
RERANKER_BATCH_SIZE_CPU = 64         # было 32
