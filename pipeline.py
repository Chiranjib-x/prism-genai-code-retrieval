"""Two-stage retrieval for CoIR AppsRetrieval, evaluated through MTEB.

Stage 1  dense retrieval. The query is an English problem statement, not
         code: older code encoders not trained for English->code search score
         poorly here (UniXcoder 1.36 NDCG@10); e5-base-v2 scores 11.52.
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

from execute import RESULTS_LOG, load_results, parse_examples, passes, result_key

EMB_CACHE = Path("cache/emb")        # content-addressed embeddings; gitignored
PASS_BOOST = 2.0                     # cosine sims live in [-1, 1]; +2 puts passers first

# e5 models were trained with these prefixes; other models get none.
PREFIXES = {"e5": ("query: ", "passage: ")}


def _prefixes(model_name: str) -> tuple[str, str]:
    return next((p for key, p in PREFIXES.items() if key in model_name.lower()), ("", ""))


def load_encoder(model_name: str, max_len: int) -> SentenceTransformer:
    """CPU encoder, set up identically for benchmark and demo."""
    # trust_remote_code: CodeXEmbed ships its model class in its HF repo.
    # Only pass model ids you have vetted.
    enc = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
    # transformers 5 loads the checkpoint's dtype; CodeXEmbed ships bfloat16,
    # which consumer CPUs emulate slowly (82 vs 619 tokens/s measured). fp32
    # is also what reproduces the model card's similarities exactly.
    enc.float()
    repair_nonpersistent_buffers(enc)
    # CPU attention is quadratic: an 8k-token context overflows 16 GB RAM and
    # thrashes swap. 1024 keeps the full text of 98% of queries, 99% of docs.
    enc.max_seq_length = min(enc.max_seq_length, max_len)
    return enc


def emb_path(model_name: str, max_len: int, texts: list[str]) -> Path:
    """Legacy whole-list cache file (pre-EmbeddingStore); kept only for migration."""
    blob = "\0".join([model_name, str(max_len), *texts]).encode()
    return EMB_CACHE / f"{hashlib.sha1(blob).hexdigest()[:16]}.npy"


class EmbeddingStore:
    """Content-addressed embeddings: one row per distinct text, per model+truncation.

    This is what makes re-indexing a new code version cheap (goal P1): only
    snippets whose text changed are encoded; unchanged ones are looked up.
    """

    def __init__(self, model_name: str, max_len: int):
        tag = hashlib.sha1(f"{model_name}|{max_len}".encode()).hexdigest()[:12]
        self.keys_path = EMB_CACHE / f"store_{tag}.keys.npy"
        self.vecs_path = EMB_CACHE / f"store_{tag}.vecs.npy"
        self.row: dict[str, int] = {}
        self.vecs: np.ndarray | None = None
        if self.keys_path.exists():
            self.row = {k: i for i, k in enumerate(np.load(self.keys_path).tolist())}
            self.vecs = np.load(self.vecs_path)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def missing(self, texts: list[str]) -> list[str]:
        return [t for t in dict.fromkeys(texts) if self.key(t) not in self.row]

    def add(self, texts: list[str], vecs: np.ndarray) -> None:
        """Append rows for texts not already stored (first occurrence wins, so
        duplicate snippets in a corpus cannot misalign rows)."""
        take: dict[str, int] = {}
        for i, t in enumerate(texts):
            k = self.key(t)
            if k not in self.row and k not in take:
                take[k] = i
        start = 0 if self.vecs is None else len(self.vecs)
        for n, k in enumerate(take):
            self.row[k] = start + n
        new = vecs[list(take.values())]
        self.vecs = new if self.vecs is None else np.vstack([self.vecs, new])
        EMB_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(self.keys_path, np.array(list(self.row)))
        np.save(self.vecs_path, self.vecs)

    def get(self, texts: list[str]) -> np.ndarray:
        """Rows for these texts, in order. KeyError if any was never encoded."""
        return self.vecs[[self.row[self.key(t)] for t in texts]]


def _extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    """transformers 4.x ModuleUtilsMixin.get_extended_attention_mask (encoder case),
    removed in 5.x but still called by CodeXEmbed's remote code."""
    dtype = dtype or torch.get_default_dtype()
    ext = attention_mask[:, None, :, :] if attention_mask.dim() == 3 else attention_mask[:, None, None, :]
    return (1.0 - ext.to(dtype)) * torch.finfo(dtype).min


def repair_nonpersistent_buffers(model: torch.nn.Module) -> None:
    """Rebuild buffers that transformers 5.x leaves uninitialised on load.

    CodeXEmbed uses Alibaba's GTE code ("new-impl"), which registers
    position_ids and the rotary inv_freq/cos/sin caches with persistent=False.
    Those are not in the checkpoint and transformers 5 no longer re-runs the
    module __init__ that fills them, so they hold garbage. position_ids crashes
    loudly; garbage cos/sin tables would silently produce wrong embeddings.
    Recomputed here exactly as the modules' own __init__ does.
    """
    for m in model.modules():
        if type(m).__name__ == "NewModel" and not hasattr(m, "get_extended_attention_mask"):
            type(m).get_extended_attention_mask = _extended_attention_mask
        pos = getattr(m, "position_ids", None)
        if isinstance(pos, torch.Tensor) and pos.dim() == 1:
            m.position_ids = torch.arange(pos.numel(), device=pos.device)
        if "RotaryEmbedding" in type(m).__name__:
            m.inv_freq = 1.0 / (m.base ** (torch.arange(0, m.dim, 2).float() / m.dim))
            seq_len = m.max_position_embeddings * getattr(m, "scaling_factor", 1.0)
            m._set_cos_sin_cache(seq_len, m.inv_freq.device, torch.get_default_dtype())


class ExecRerankSearch:
    """Dense retrieval + execution-verified rerank, as an MTEB SearchProtocol."""

    def __init__(self, model_name: str, k: int, rerank: bool, workers: int,
                 max_queries: int = 0, max_len: int = 1024):
        self.model_name, self.k, self.rerank, self.workers = model_name, k, rerank, workers
        self.max_queries = max_queries  # 0 = no limit
        self.encoder = load_encoder(model_name, max_len)
        print(f"{model_name}: max_seq_length={self.encoder.max_seq_length}", flush=True)
        self.q_prefix, self.d_prefix = _prefixes(model_name)
        self.cache: dict[str, bool] = load_results()
        self.stats: dict[str, float] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Normalised embeddings; only texts never seen before are encoded."""
        store = EmbeddingStore(self.model_name, self.encoder.max_seq_length)
        new = store.missing(texts)
        print(f"encode: {len(texts):,} texts, {len(new):,} not yet indexed", flush=True)
        if new:
            store.add(new, self.encoder.encode(new, batch_size=16, normalize_embeddings=True,
                                               show_progress_bar=True, convert_to_numpy=True))
        return store.get(texts)

    # -- MTEB SearchProtocol ------------------------------------------------

    @property
    def mteb_model_meta(self) -> ModelMeta:
        suffix = f"exec-k{self.k}" if self.rerank else "dense"
        if self.max_queries:
            # Sliced runs still score against every qrel, so the numbers are
            # deflated. Never let one be mistaken for the submission file.
            suffix += f"-smoke{self.max_queries}"
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
        self.doc_emb = self._encode([self.d_prefix + t for t in corpus["text"]])
        self.stats["index_s"] = time.time() - t0

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs,
               top_ranked=None, num_proc=None) -> dict[str, dict[str, float]]:
        t0 = time.time()
        qids, qtexts = list(queries["id"]), list(queries["text"])
        if self.max_queries:
            qids, qtexts = qids[: self.max_queries], qtexts[: self.max_queries]
        q_emb = self._encode([self.q_prefix + t for t in qtexts])
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
            ranked = sorted(results[qid], key=results[qid].get, reverse=True)[: self.k]
            jobs += [(qid, d, result_key(self.doc_text[d], examples), examples) for d in ranked]

        # one run per distinct (program content, examples) -- duplicates share it
        todo = list({j[2]: j for j in jobs if j[2] not in self.cache}.values())
        print(f"execution: {len(jobs)} candidate checks, {len(todo)} uncached", flush=True)
        RESULTS_LOG.parent.mkdir(exist_ok=True)
        t0 = time.time()
        with RESULTS_LOG.open("a", encoding="utf-8") as log, \
                ThreadPoolExecutor(self.workers) as pool:
            results_iter = pool.map(lambda j: passes(self.doc_text[j[1]], j[3]), todo)
            for n, ((_, _, key, _), ok) in enumerate(zip(todo, results_iter), 1):
                self.cache[key] = ok
                log.write(f"{key}\t{int(ok)}\n")
                if n % 5000 == 0:
                    log.flush()
                    rate = n / (time.time() - t0)
                    print(f"  {n:,}/{len(todo):,} checked  {rate:.0f}/s  "
                          f"eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)

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
    ap.add_argument("--max-len", type=int, default=1024, help="token cap per text (CPU RAM)")
    args = ap.parse_args()

    model = ExecRerankSearch(args.model, args.k, not args.no_rerank, args.workers,
                             args.max_queries, args.max_len)
    name = model.mteb_model_meta.name.split("/")[-1]
    result = mteb.evaluate(model, [mteb.get_task("AppsRetrieval")], cache=None,
                           prediction_folder=Path("results/predictions") / name)

    task_result = list(result.task_results)[0]
    out = Path("results") / f"appsretrieval_{name}.json"
    out.parent.mkdir(exist_ok=True)
    # default=str: current MTEB puts a datetime in to_dict(), which plain json.dumps rejects.
    out.write_text(json.dumps(task_result.to_dict(), indent=2, default=str))

    s = task_result.to_dict()["scores"]["test"][0]
    print(f"\n{name}")
    print(f"  ndcg@10 {s['ndcg_at_10'] * 100:6.2f}   mrr@10 {s['mrr_at_10'] * 100:6.2f}   "
          f"recall@{args.k} {s.get(f'recall_at_{args.k}', float('nan')) * 100:6.2f}   "
          f"recall@100 {s['recall_at_100'] * 100:6.2f}")
    print("  " + "   ".join(f"{k} {v:.1f}" for k, v in model.stats.items()))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
