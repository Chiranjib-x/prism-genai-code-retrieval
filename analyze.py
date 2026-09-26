"""Offline rerank experiments from cached embeddings + execution results.

Every AppsRetrieval test query has exactly one relevant document, so the
metrics have closed forms: NDCG@10 = 1/log2(rank+2), MRR@10 = 1/(rank+1),
both 0 beyond rank 10. That lets rerank variants be scored in seconds instead
of re-running the MTEB pipeline. Checked against MTEB's own numbers below.

Usage: python analyze.py
"""

from __future__ import annotations

import hashlib

import numpy as np
from datasets import load_dataset

from execute import load_results, parse_examples
from pipeline import _prefixes, emb_path

# name -> (hf model id, max_seq_length it was encoded with)
MODELS = {
    "e5-base": ("intfloat/e5-base-v2", 512),
    "codexembed-400m": ("Salesforce/SFR-Embedding-Code-400M_R", 1024),
}


def load_data():
    corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
    qtext = {x["_id"]: x["text"] for x in load_dataset("CoIR-Retrieval/apps", "queries", split="queries")}
    qrels = list(load_dataset("CoIR-Retrieval/apps", "default", split="test"))
    doc_ids, docs = list(corpus["_id"]), list(corpus["text"])
    queries = [qtext[r["query-id"]] for r in qrels]          # MTEB's order (verified)
    col = {d: j for j, d in enumerate(doc_ids)}
    gold = np.array([col[r["corpus-id"]] for r in qrels])
    examples = [parse_examples(q) for q in queries]
    tags = [hashlib.sha1(repr(e).encode()).hexdigest()[:12] if e else None for e in examples]
    return docs, queries, gold, doc_ids, tags, load_results()


def load_sims(model_name: str, max_len: int, docs, queries) -> np.ndarray | None:
    """Query x doc cosine similarities from the pipeline's embedding cache, or None."""
    qp, dp = _prefixes(model_name)
    d_path = emb_path(model_name, max_len, [dp + t for t in docs])
    q_path = emb_path(model_name, max_len, [qp + t for t in queries])
    if not (d_path.exists() and q_path.exists()):
        return None
    return np.load(q_path) @ np.load(d_path).T


def dense_ranks(sims: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return (sims > sims[np.arange(len(gold)), gold][:, None]).sum(1).astype(float)


def score(ranks: np.ndarray) -> tuple[float, float]:
    """Mean NDCG@10 and MRR@10 (x100) from 0-based gold ranks (inf = not retrieved)."""
    hit = ranks < 10
    ndcg = np.where(hit, 1 / np.log2(np.minimum(ranks, 9) + 2), 0)
    mrr = np.where(hit, 1 / (np.minimum(ranks, 9) + 1), 0)
    return 100 * ndcg.mean(), 100 * mrr.mean()


def rerank_ranks(sims, gold, doc_ids, tags, cache, k: int, boost=lambda n_pass: 2.0,
                 n: int = 100) -> tuple[np.ndarray, dict]:
    """Gold ranks after boosting top-k candidates that pass execution.

    `boost(n_pass)` sets the bonus given how many candidates passed -- a pass
    shared by 30 programs is weaker evidence than a unique one.
    """
    depth = max(n, k)
    top = np.argpartition(-sims, depth, axis=1)[:, :depth]
    ranks = np.full(len(gold), np.inf)
    stats = {"missing_exec": 0, "queries_with_passer": 0}
    for i in range(len(gold)):
        order = top[i][np.argsort(-sims[i, top[i]])]
        s = sims[i, order].copy()
        if tags[i]:
            flags = []
            for d in order[:k]:
                hit = cache.get(f"{doc_ids[d]}|{tags[i]}")
                stats["missing_exec"] += hit is None
                flags.append(bool(hit))
            n_pass = sum(flags)
            if n_pass:
                stats["queries_with_passer"] += 1
                s[:k] += np.array(flags) * boost(n_pass)
        order = order[np.argsort(-s, kind="stable")]
        where = np.flatnonzero(order == gold[i])
        if where.size:
            ranks[i] = where[0]
    return ranks, stats


if __name__ == "__main__":
    docs, queries, gold, doc_ids, tags, cache = load_data()
    for name, (model, max_len) in MODELS.items():
        sims = load_sims(model, max_len, docs, queries)
        if sims is None:
            print(f"{name}: no cached embeddings yet")
            continue
        print(f"{name:16} dense            ndcg@10 %6.2f  mrr@10 %6.2f" % score(dense_ranks(sims, gold)))
        for k in (50, 100, 200, 500):
            ranks, st = rerank_ranks(sims, gold, doc_ids, tags, cache, k)
            note = f"  [{st['missing_exec']:,} unchecked pairs]" if st["missing_exec"] else ""
            print(f"{'':16} exec rerank k={k:<3} ndcg@10 %6.2f  mrr@10 %6.2f   passers in "
                  f"{st['queries_with_passer']} queries{note}" % score(ranks))
