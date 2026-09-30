# Execution-verified code retrieval on CPU

Samsung PRISM GenAI Hackathon 2026 · Theme 1: Agentic Code Intelligence

Given a programming problem written in plain English, find the Python program
that solves it, among 8,765 candidates, on a CPU.

## Approach

**Stage 1: dense retrieval.** The problem statement and every program are
embedded; candidates are ranked by cosine similarity. Code-retrieval embedders
trained for English→code search do far better here than general text models
(see results).

**Stage 2: execution verification.** Competitive-programming statements contain
worked examples ("Input … Output …"). We parse them and *run* the top-k
candidates on each example in a sandboxed subprocess. Programs that reproduce
every expected output move to the front; similarity order is kept within each
group. Relevance stops being estimated and becomes checked.

Both stages run through MTEB's `SearchProtocol`, so one command produces the
official results JSON.

## Results — CoIR `AppsRetrieval`, full test split (3,765 queries)

| System | NDCG@10 | MRR@10 |
|---|---:|---:|
| BM25 (published) | 0.95 | — |
| e5-base-v2, dense (reproduced; published 11.52) | 11.52 | 9.88 |
| e5-base-v2 + execution rerank, top-50 | 20.71 | 19.96 |
| CodeXEmbed-400M, dense | *pending* | *pending* |
| CodeXEmbed-400M + execution rerank | *pending* | *pending* |

Full ablations, recall-by-depth and negative results: [RESULTS.md](RESULTS.md).

Official MTEB result: [results/appsretrieval_e5-base-v2-exec-k50.json](results/appsretrieval_e5-base-v2-exec-k50.json).
Per-query rankings (top 1,000 for every query) are gzipped to stay under GitHub's
file limit: `results/predictions/e5-base-v2-exec-k50/AppsRetrieval_predictions.json.gz`.

## Submission materials

- Presentation: [PDF](VITV_Error404_1_Presentation.pdf) · [PPTX](VITV_Error404_1_Presentation.pptx)
- Demo video: https://youtu.be/WuWeffTKWeM
- AI disclosure: [AI_DISCLOSURE.md](AI_DISCLOSURE.md)

## Setup

Python 3.11, CPU only.

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Or with Docker: see the header of [Dockerfile](Dockerfile).

## Run

```bash
# Full benchmark (submitted config: e5-base-v2, top-50 execution rerank).
# Writes results/appsretrieval_<config>.json, the MTEB submission file.
python pipeline.py

# Stage 1 only (ablation)
python pipeline.py --no-rerank

# Other encoders, e.g. CodeXEmbed-400M (integrated; full run not yet completed)
python pipeline.py --model Salesforce/SFR-Embedding-Code-400M_R --k 100

# Interactive demo: ranked solutions + timings for one problem
python search.py --qid q5001
python search.py --file my_problem.txt

# Offline ablations from cached embeddings and execution results (seconds)
python analyze.py

# Verifier self-test
python execute.py
```

The first run downloads the model and dataset and encodes the corpus. Later
runs reuse the embedding store and execution cache in `cache/`.

## Retrieval across code versions (P1)

Both indexes are **content-addressed**:

- `EmbeddingStore` keeps one vector per distinct snippet text. Indexing a new
  version of the corpus encodes only snippets whose text changed.
- The execution cache is keyed by *program content + examples*, never by
  document id, so an edited snippet cannot be served a stale pass/fail.

## Sandbox

Candidate programs come from a public dataset and are treated as untrusted.
Each runs as a fresh `python -I` subprocess (isolated mode) in a throwaway
temporary directory, with a hard timeout and a stripped environment. The
Windows build has no memory or network limits at the process level; in Docker,
run with `--network none` and `--memory` once models are cached.

## Engineering notes

Things that silently break this pipeline on current library versions, each
fixed and commented in the code:

- **Import order on Windows**: `torch` must be imported before `mteb`, or
  `c10.dll` fails to initialise (WinError 1114).
- **MTEB JSON export**: `TaskResult.to_dict()` contains a `datetime`; plain
  `json.dumps` raises. We serialise with `default=str`.
- **CodeXEmbed on transformers 5**: its remote code calls a method removed in
  v5 and relies on non-persistent buffers v5 leaves uninitialised. We restore
  both; the output matches the model card's reference similarities exactly.
- **bfloat16 on CPU**: transformers 5 keeps the checkpoint dtype. CodeXEmbed
  ships bf16, which laptop CPUs emulate: 82 tokens/s vs 619 in float32.
- **Memory**: an 8k-token context overflows 16 GB RAM and swaps. We cap at
  1,024 tokens, which keeps the full text of 98% of queries and 99% of programs.
- **Timeouts under load**: running execution while an encoder saturates the CPU
  causes spurious timeouts, cached as false failures. Stages run sequentially.

## Prior art

Execution-based reranking of retrieved code on APPS is also studied in
*ExecRetrieval* (arXiv 2609.01865). Our contribution is a CPU-only pipeline
combining a code-retrieval embedder with sandboxed execution verification,
evaluated end to end through MTEB, with content-addressed indexes for
versioned corpora.

## Licences

Code in this repository: MIT (see [LICENSE](LICENSE)). The stage-1 model
`Salesforce/SFR-Embedding-Code-400M_R` is **CC-BY-NC-4.0**
(non-commercial).
