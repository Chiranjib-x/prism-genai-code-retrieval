# Results log — CoIR AppsRetrieval (test split, 3,765 queries)

All numbers are full test split, CPU only. NDCG@10 / MRR@10 ×100. "Offline" rows
come from `analyze.py`, which reproduces MTEB's scores exactly (verified below).

## Main table

| # | System | NDCG@10 | MRR@10 | Source |
|---|---|---:|---:|---|
| — | BM25 (published, CoIR paper) | 0.95 | — | literature |
| — | UniXcoder, code-specific (published) | 1.36 | — | literature |
| — | E5-Mistral-7B (published) | 21.33 | — | literature |
| — | Best reported on APPS (CoIR paper) | 26.52 | — | literature |
| 1 | e5-base-v2, dense only | **11.52** | **9.88** | MTEB, matches published exactly |
| 2 | + execution rerank, top-50 | **20.71** | **19.96** | MTEB |

Row 2 is level with E5-Mistral-7B using a model ~65× smaller, on CPU.

## Why execution depth matters (e5-base-v2 dense recall)

Execution can only rescue a gold solution that stage 1 retrieved.

| Depth K | Recall, all queries | Recall, queries with examples | Execution checks |
|---:|---:|---:|---:|
| 10 | 16.9% | 13.7% | 29,360 |
| 50 | 28.0% | 24.1% | 146,800 |
| 100 | 34.3% | 30.5% | 293,600 |
| 200 | 42.3% | 38.4% | 587,200 |
| 500 | 56.1% | 52.5% | 1,468,000 |
| 1000 | 68.1% | 64.8% | 2,936,000 |

## Data facts (measured)

- 2,936 / 3,765 test queries (78%) contain parseable worked examples.
- Gold solutions pass their own examples 85% of the time (300-query sample).
  Failures: "print any valid answer" problems, and broken dataset entries
  (e.g. `SyntaxError: 'return' outside function`).
- At top-50, 628 queries have at least one passing candidate.
- Throughput ~263 candidate checks/s (24 workers, Windows process spawn).

## Negative / cautionary results

- The published 30.74 NDCG@10 for execution rerank came from a 150-query
  sample. At full scale with the same base model and K=50 we measure 20.71.
- Spurious timeouts occur when execution shares CPU with an encoder. Stages
  must run sequentially.

## Timings (e5-base-v2, 28-core CPU)

- Corpus encoding (8,765 docs): ~12 min · query encoding (3,765): ~11 min
- Execution, 146,800 checks: ~9 min
