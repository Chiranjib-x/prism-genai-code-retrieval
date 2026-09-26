"""Retrieval demo: a problem statement in, ranked solutions out, with timings.

  python search.py --qid q5001                 # a test-split query; shows where gold lands
  python search.py --file problem.txt          # any problem statement
  python search.py "Given n integers, print the largest..."

Stage 1 ranks the whole corpus by embedding similarity. Stage 2 runs the top-k
candidates on the worked examples in the statement; programs that reproduce
every example move to the front. Uses the same model setup and scoring as the
MTEB benchmark (pipeline.py). Execution results here are not written to the
shared cache, so a demo on a busy machine cannot record spurious timeouts.
"""

from __future__ import annotations

import torch  # noqa: F401  -- must precede mteb (imported via pipeline) on Windows

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from datasets import load_dataset

from execute import parse_examples, passes
from pipeline import PASS_BOOST, _prefixes, emb_path, load_encoder

DEFAULT_MODEL = "Salesforce/SFR-Embedding-Code-400M_R"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="problem statement text")
    ap.add_argument("--qid", help="use a CoIR apps test query, e.g. q5001")
    ap.add_argument("--file", help="read the problem statement from a file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--k", type=int, default=100, help="candidates checked by execution")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
    doc_ids, docs = list(corpus["_id"]), list(corpus["text"])

    gold = None
    if args.qid:
        qtext = {x["_id"]: x["text"] for x in load_dataset("CoIR-Retrieval/apps", "queries", split="queries")}
        gold = {r["query-id"]: r["corpus-id"] for r in load_dataset("CoIR-Retrieval/apps", "default", split="test")}.get(args.qid)
        query = qtext[args.qid]
    elif args.file:
        query = Path(args.file).read_text(encoding="utf-8")
    elif args.query:
        query = args.query
    else:
        ap.error("give a query, --qid or --file")

    t = time.perf_counter()
    enc = load_encoder(args.model, args.max_len)
    q_prefix, d_prefix = _prefixes(args.model)
    path = emb_path(args.model, enc.max_seq_length, [d_prefix + d for d in docs])
    if not path.exists():
        raise SystemExit(f"No corpus index for {args.model} at max_len {enc.max_seq_length}.\n"
                         f"Build it once with: python pipeline.py --model {args.model} --no-rerank")
    index = np.load(path)
    t_load = time.perf_counter() - t

    t = time.perf_counter()
    q = enc.encode([q_prefix + query], normalize_embeddings=True)[0]
    t_encode = time.perf_counter() - t

    t = time.perf_counter()
    sims = index @ q
    top = np.argsort(-sims)[: args.k]
    t_search = time.perf_counter() - t

    examples = parse_examples(query)
    t = time.perf_counter()
    if examples:
        with ThreadPoolExecutor(24) as pool:
            ok = np.array(list(pool.map(lambda j: passes(docs[j], examples), top)))
    else:
        ok = np.zeros(len(top), dtype=bool)
    t_exec = time.perf_counter() - t

    order = top[np.argsort(-(sims[top] + PASS_BOOST * ok), kind="stable")]
    passed = dict(zip(top.tolist(), ok.tolist()))

    print(f"\nquery: {query.strip().splitlines()[0][:100]}")
    print(f"worked examples found: {len(examples)}   candidates executed: {len(top) if examples else 0}"
          f"   passed: {int(ok.sum())}\n")
    print(f"{'rank':>4}  {'doc':<7} {'sim':>6}  {'exec':<5} first line")
    for r, j in enumerate(order[: args.top], 1):
        mark = ("PASS" if passed[j] else "fail") if examples else "-"
        first = next((ln.strip() for ln in docs[j].splitlines() if ln.strip()), "")[:60]
        star = "  <-- gold" if doc_ids[j] == gold else ""
        print(f"{r:>4}  {doc_ids[j]:<7} {sims[j]:6.3f}  {mark:<5} {first}{star}")

    if gold:
        g = doc_ids.index(gold)
        dense_rank = int((sims > sims[g]).sum()) + 1
        final = np.flatnonzero(order == g)
        final_rank = int(final[0]) + 1 if final.size else None
        print(f"\ngold {gold}: dense rank {dense_rank} -> final rank "
              f"{final_rank if final_rank else f'>{args.k} (not in candidates)'}")

    print(f"\ntimings: model+index load {t_load:.1f}s | query encode {1000 * t_encode:.0f} ms | "
          f"search {1000 * t_search:.1f} ms over {len(docs):,} docs | execution {1000 * t_exec:.0f} ms")


if __name__ == "__main__":
    main()
