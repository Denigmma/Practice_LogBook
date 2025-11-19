# src/main.py
import sys
import json
import time

from . import config
from .preprocessor import load_queries
from .retriever import AlfabankRetrieval

def main() -> None:
    print("PROJECT_ROOT =", config.PROJECT_ROOT, flush=True)
    print("WEBSITES_CSV =", config.WEBSITES_CSV, flush=True)
    print("QUESTIONS_CSV =", config.QUESTIONS_CSV, flush=True)
    print("CHROMA_DIR   =", config.CHROMA_DIR, flush=True)

    retr = AlfabankRetrieval()

    t0 = time.time()
    retr.build_index()
    print(f"[INFO] Index ready in {time.time()-t0:.1f}s. Loading queries...", flush=True)

    dfq = load_queries()
    print(f"[INFO] Queries loaded: {len(dfq)}. Starting answering...", flush=True)

    t1 = time.time()
    out = retr.batch_answer(dfq)
    print(f"[INFO] Answering done in {time.time()-t1:.1f}s. Saving submit.csv ...", flush=True)

    # web_list — строка вида "[1,2,3,4,5]"
    out["web_list"] = out["web_list"].apply(lambda xs: json.dumps(xs, ensure_ascii=False))
    out.to_csv("submit.csv", index=False)
    print("submit.csv saved", flush=True)

    retr.close()
    print("[INFO] Done. Exiting.", flush=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.exit(0)
