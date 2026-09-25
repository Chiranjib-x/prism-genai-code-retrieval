"""Two-stage retrieval for CoIR AppsRetrieval, evaluated through MTEB.

Stage 1  dense retrieval with a *general text* embedder. The query is an
         English problem statement, not code -- code-specific encoders score
         an order of magnitude worse on this dataset (UniXcoder 1.36 NDCG@10).
Stage 2  execution verification. Candidates in the top-K that reproduce the
         query's worked examples are moved above those that do not; dense
         order is kept within each group.

Implements MTEB's SearchProtocol, so `mteb.evaluate` scores the full pipeline
and writes the official results JSON.

Usage:
  python pipeline.py                         # e5-base-v2, K=50, execution rerank
  python pipeline.py --no-rerank             # stage 1 only (ablation arm)
  python pipeline.py --model BAAI/bge-base-en-v1.5 --k 100
"""

from __future__ import annotations

import torch  # noqa: F401  -- must precede mteb on Windows (c10.dll load order)

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mteb
import numpy as np
from mteb.models.model_meta import ModelMeta
from sentence_transformers import SentenceTransformer

from execute import parse_examples, passes

CACHE = Path("cache/exec.json")      # (doc, examples) -> pass/fail; gitignored
PASS_BOOST = 2.0                     # cosine sims live in [-1, 1]; +2 puts passers first

# e5 models were trained with these prefixes; other models get none.
PREFIXES = {"e5": ("query: ", "passage: ")}


def _prefixes(model_name: str) -> tuple[str, str]:
    return next((p for key, p in PREFIXES.items() if key in model_name.lower()), ("", ""))


class ExecRerankSearch:
    """Dense retrieval + execution-verified rerank, as an MTEB SearchProtocol."""

    def __init__(self, model_name: str, k: int, rerank: bool, workers: int, max_queries: int = 0):
        self.model_name, self.k, self.rerank, self.workers = model_name, k, rerank, workers
        self.max_queries = max_queries  # 0 = no limit
        self.encoder = SentenceTransformer(model_name, device="cpu")
        self.q_prefix, self.d_prefix = _prefixes(model_name)
        self.cache: dict[str, bool] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self.stats: dict[str, float] = {}

    # -- MTEB SearchProtocol ------------------------------------------------

    @property
    def mteb_model_meta(self) -> ModelMeta:
        suffix = f"exec-k{self.k}" if self.rerank else "dense"
        return ModelMeta(
            loader=None, name=f"local/{self.model_name.split('/')[-1]}-{suffix}",
            revision="1", release_date="2026-09-26", languages=["eng-Latn", "python-Code"],
            n_parameters=None, memory_usage_mb=None, max_tokens=512, embed_dim=None,
            license=None, open_weights=True, public_training_code=None,
            public_training_data=None, framework=["Sentence Transformers"],
            similarity_fn_name="cosine", use_instructions=False, training_datasets=None,
        )

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs,
              num_proc=None) -> None:
        t0 = time.time()
        self.doc_ids = list(corpus["id"])
        self.doc_text = dict(zip(self.doc_ids, corpus["text"]))
        self.doc_emb = self.encoder.encode(
            [self.d_prefix + t for t in corpus["text"]], batch_size=32,
            normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        self.stats["index_s"] = time.time() - t0

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs,
               top_ranked=None, num_proc=None) -> dict[str, dict[str, float]]:
        t0 = time.time()
        qids, qtexts = list(queries["id"]), list(queries["text"])
        if self.max_queries:
            qids, qtexts = qids[: self.max_queries], qtexts[: self.max_queries]
        q_emb = self.encoder.encode(
            [self.q_prefix + t for t in qtexts], batch_size=32,
            normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        sims = q_emb @ self.doc_emb.T
        n = max(top_k, self.k)
        top = np.argpartition(-sims, n, axis=1)[:, :n]
        self.stats["dense_s"] = time.time() - t0

        results: dict[str, dict[str, float]] = {}
        for i, qid in enumerate(qids):
            idx = top[i][np.argsort(-sims[i, top[i]])]
            results[qid] = {self.doc_ids[j]: float(sims[i, j]) for j in idx}

        if self.rerank:
            t1 = time.time()
            self._execution_rerank(qids, qtexts, results)
            self.stats["exec_s"] = time.time() - t1
        return results

    # -- stage 2 ------------------------------------------------------------

    def _execution_rerank(self, qids, qtexts, results) -> None:
        jobs = []                                    # (qid, doc_id, cache key, examples)
        for qid, text in zip(qids, qtexts):
            examples = parse_examples(text)
            if not examples:
                continue
            tag = hashlib.sha1(repr(examples).encode()).hexdigest()[:12]
            ranked = sorted(results[qid], key=results[qid].get, reverse=True)[: self.k]
            jobs += [(qid, d, f"{d}|{tag}", examples) for d in ranked]

        todo = [j for j in jobs if j[2] not in self.cache]
        print(f"execution: {len(jobs)} candidate checks, {len(todo)} uncached")
        with ThreadPoolExecutor(self.workers) as pool:
            for (_, doc, key, examples), ok in zip(
                    todo, pool.map(lambda j: passes(self.doc_text[j[1]], j[3]), todo)):
                self.cache[key] = ok
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(self.cache))

        boosted = set()
        for qid, doc, key, _ in jobs:
            if self.cache[key]:
                results[qid][doc] += PASS_BOOST
                boosted.add(qid)
        self.stats["queries_with_examples"] = len({j[0] for j in jobs})
        self.stats["queries_with_a_passer"] = len(boosted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--k", type=int, default=50, help="candidates checked by execution")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-queries", type=int, default=0, help="slice queries for smoke tests (0=all)")
    args = ap.parse_args()

    model = ExecRerankSearch(args.model, args.k, not args.no_rerank, args.workers, args.max_queries)
    name = model.mteb_model_meta.name.split("/")[-1]
    result = mteb.evaluate(model, [mteb.get_task("AppsRetrieval")], cache=None,
                           prediction_folder=Path("results/predictions") / name)

    task_result = list(result.task_results)[0]
    out = Path("results") / f"appsretrieval_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(task_result.to_dict(), indent=2))

    s = task_result.to_dict()["scores"]["test"][0]
    print(f"\n{name}")
    print(f"  ndcg@10 {s['ndcg_at_10'] * 100:6.2f}   mrr@10 {s['mrr_at_10'] * 100:6.2f}   "
          f"recall@{args.k} {s.get(f'recall_at_{args.k}', float('nan')) * 100:6.2f}   "
          f"recall@100 {s['recall_at_100'] * 100:6.2f}")
    print("  " + "   ".join(f"{k} {v:.1f}" for k, v in model.stats.items()))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
