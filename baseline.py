"""Stage-1 baseline: dense retrieval on CoIR AppsRetrieval via MTEB.

Usage: python baseline.py [hf-model-id]    (default: intfloat/e5-base-v2)

Uses MTEB's registered model wrapper so query/passage prompts match the
model's training. Published reference for e5-base-v2: NDCG@10 ~= 11.52.
"""

import json
import sys
import time
from pathlib import Path

# Must precede mteb on Windows: mteb loads another library's DLLs first, and
# torch's c10.dll then fails to initialise (WinError 1114).
import torch  # noqa: F401

import mteb

MODEL = sys.argv[1] if len(sys.argv) > 1 else "intfloat/e5-base-v2"
OUT = Path("results")


def main() -> None:
    model = mteb.get_model(MODEL)
    task = mteb.get_task("AppsRetrieval")

    t0 = time.time()
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})
    elapsed = time.time() - t0

    task_result = list(result.task_results)[0]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"appsretrieval_{MODEL.replace('/', '__')}.json"
    path.write_text(json.dumps(task_result.to_dict(), indent=2))

    scores = task_result.to_dict()["scores"]["test"][0]
    print(f"model      {MODEL}")
    print(f"ndcg@10    {scores['ndcg_at_10'] * 100:.2f}")
    print(f"mrr@10     {scores['mrr_at_10'] * 100:.2f}")
    print(f"recall@100 {scores['recall_at_100'] * 100:.2f}")
    print(f"elapsed    {elapsed / 60:.1f} min")
    print(f"saved      {path}")


if __name__ == "__main__":
    main()
