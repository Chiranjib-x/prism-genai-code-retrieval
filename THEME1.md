# Theme 01 — Agentic Code Intelligence: analysis and plan

Written 24 Sep 2026. Deadline **30 Sep, 11:59 PM** — 6 days.

---

## 1. What is actually being scored

Strip the theme title away. The screening is one number on one public dataset:

- **Task**: given a natural-language query, rank code snippets by relevance. Retrieval only —
  generation, answering, and explanation are **explicitly out of scope**.
- **Dataset**: CoIR `apps` test split (`mteb.get_task("AppsRetrieval")`).
  ~8.8k corpus docs, **3,765 test queries**. Queries are competitive-programming problem
  statements in English; the corpus is **Python** solution code.
- **Metrics**: NDCG@10 and MRR.
- **Deliverable**: the MTEB-generated JSON, uploaded as a **GitHub release artifact**.
- **Constraint**: must run on CPU, minimal GPU.
- **Then**: competitive screening → top submissions go to hands-on (PPT, demo video, live
  queries, plus P1 and the Bonus).

⚠️ **Note the contradiction**: the main hackathon deck says Theme 1 is "Single language:
JavaScript". The theme-1 guidelines PDF says screening is CoIR apps, which is **Python**.
The guidelines are the operative document for the number we are ranked on. Build for CoIR
apps; treat any JavaScript codebase as a hands-on-round concern.

## 2. Why this dataset is brutal — and why that is good news

APPS is **the hardest of CoIR's ten datasets by a wide margin.** Published NDCG@10:

| Method | NDCG@10 on APPS |
|---|---:|
| BM25 (lexical) | **0.95** |
| UniXcoder (code-specific) | **1.36** |
| E5-Mistral-7B | 21.33 |
| Best reported (CoIR paper) | **26.52** |

Models that score 50–70 on the other nine CoIR datasets **collapse to single digits here.**

The reason is a pure modality gap. The query describes *a task* ("given N integers, find the
longest subsequence such that…"); the document is *a solution* — variable names, loops, no
restatement of the problem. Lexical overlap is close to zero, which is why BM25 scores 0.95:
**statistically indistinguishable from random.**

This is good news for us. A near-zero floor and a 26.52 ceiling means the field is wide open
and small ideas move the number a lot. On an easy dataset everyone clusters at 70 and you win
by 0.3 points. Here there is room to win by ten.

## 3. The two findings that decide our approach

### Finding A — code-specific embedders are the wrong tool

Counterintuitive and well documented: **UniXcoder (code-specific) scores 1.36; E5-base-v2 (a
general text model) scores 11.52.** An 8× gap in favour of the *non*-code model.

The reason, once seen, is obvious: **the query is not code.** It is English prose with maths
and worked examples. You are embedding a problem statement, and a model trained to encode
syntax has nothing useful to say about it. Most teams will reach for a code embedder because
the theme says "code intelligence". That instinct costs them an order of magnitude.

### Finding B — the unlock is execution, not embedding

APPS problem statements **contain worked example inputs and outputs.** That means relevance is
not only estimable — it is **checkable**. Parse the examples out of the query, run each
candidate snippet as a subprocess against them, and see which actually produce the expected
output.

Published CPU-only result on this exact benchmark:

| Stage | NDCG@10 | MRR |
|---|---:|---:|
| E5-base-v2 dense retrieval alone | 11.52 | 9.88 |
| **+ execution verification rerank** | **30.74** | **30.28** |

**30.74 beats the 26.52 reported state of the art**, on CPU, with a general-purpose base model.

Supporting stats from the same run (K=50, all 3,765 queries, 143 min on 2 cores):
- 924 queries (24.5%) had at least one passing snippet
- where something passed, the gold snippet was among the passers **60%** of the time
- the gold snippet was the **only** passer in **49%** of cases

That last number is the interesting one. When execution fires, it is usually *decisive* — not
a weak signal to blend, but a near-oracle for half the cases it touches.

**Caveats we must not paper over:** the 30.74 figure is from a **150-query sample**, not the
full test split. It must be reproduced at full scale before it goes anywhere near a slide.

## 4. Where the headroom is

Two levers, and they multiply rather than add.

**Lever A — Recall@50 is 31.29%.** That is the hard ceiling on what any reranker can fix. In
69% of queries the gold snippet is not even in the candidate set, so execution cannot rescue it.
Every point of recall gained hands the reranker more to work with.

**Lever B — execution verification.** The multiplier, worth roughly 2.7× on the sample.

**And the gap between them:** execution only fires for ~25% of queries. For the other 75% the
ranking is whatever stage 1 produced. Improving the dense ranking is therefore *not* optional
polish — it decides three quarters of the score.

Concretely, in priority order:
1. Reproduce the 11.52 baseline. Nothing else is meaningful until this exists.
2. Sweep general text embedders for Recall@50 (e5-large-v2, bge-large, gte, jina-v3). Ignore
   code models except as one documented negative result — which is itself a good PPT slide.
3. Execution verification with a hard sandbox.
4. Query preprocessing: the guidelines explicitly invite "categorizing" and "pre-processing the
   query". Worth testing whether stripping the worked examples *before embedding* helps stage 1
   (they are noise for semantic matching) while keeping them for stage 2 (they are the whole
   signal for execution). **Same input, opposite treatment per stage** — a clean, defensible idea.
5. Hybrid fusion (dense + BM25 via RRF) — but note BM25 scores 0.95 here, so expect little.
   Cheap to test, worth one hour, likely a documented negative.

**Our biggest gain uses no neural model at all.** That is a genuinely strong story against the
CPU constraint, and it is the opposite of what every other team will present.

## 5. Risks, named

| Risk | Severity | Mitigation |
|---|---|---|
| **Executing untrusted dataset code** | **High** | Non-negotiable sandbox: subprocess, hard timeout, memory cap, no network, no filesystem writes, separate working dir. This is also a PPT slide, not just a chore. |
| 30.74 came from a 150-query sample | High | Reproduce on all 3,765 before claiming it anywhere |
| Execution runtime (143 min / 2 cores) | Medium | Budget it; parallelise across cores; cache results per (snippet, test-case) |
| Only 24.5% of queries get a passing snippet | Medium | Stage-1 quality decides the other 75% — do not over-invest in execution alone |
| P1 / Bonus are hands-on-round criteria | Medium | Need a version story, but it earns nothing in screening. Do not let it eat screening time |
| Deck says JavaScript, benchmark is Python | Low | Build for the benchmark; mention the discrepancy in the PPT |

## 6. Six-day plan (24 → 30 Sep)

| Day | Work | Exit condition |
|---|---|---|
| **24 Sep** | Env setup, load CoIR apps via MTEB, run E5-base-v2 end to end | **A real NDCG@10 on the board (~11.5)** + valid results JSON |
| 25 | Embedder sweep for Recall@50; query preprocessing experiments | Best stage-1 recall chosen and measured |
| 26 | Execution verification + sandbox; validate on 150 queries first | Sample reproduces ≥ 25 NDCG@10 |
| 27 | Full 3,765-query execution run; tune K, tie-breaks, fallback ranking | **Final number on the full test split** |
| 28 | P1 version story + incremental reindex; start PPT | Ablation table complete |
| 29 | Demo video, README, Dockerfile, AI disclosure; release tag + JSON | Fresh clone reproduces the number |
| 30 | Buffer; submit early | Form submitted well before 11:59 PM |

**Never cut:** the MTEB results JSON and the release tag — *that artifact is the screening
submission*; without it there is no ranking. Then README, PPT, demo video, AI disclosure.

**Cut order if short:** evolutionary-retrieval bonus → version story → embedder sweep breadth →
hybrid fusion. All of these are hands-on-round value, and hands-on only matters if the
screening number gets us there.

## 7. Submission mechanics (from the form)

- Team name format has **changed**: `CollegeName_Teamname_ThemeNo` → e.g. `VITV_<Team>_1`.
  Earlier docs said `CollegeName_TeamName` with no theme number. Use the form's version.
- College: **VITV**
- GitHub tag: `PRISM_GENAI_HACKATHON_Y2026`
- Repo checklist on the form: source code, presentation, video, **AI disclosure**, README,
  APK/SDK (n/a), TAG
- The AI disclosure form goes **in the repo**, with per-feature origin classification
  (tool, prompt, output summary, modifications)
- Demo must show the solution answering queries and **how fast it is** — not just numbers

## 8. Consequence for the Theme 5 work

`engine.py`, `test_engine.py` and `RESEARCH.md` are a different problem with **no reuse** here.
Interruption-safe agent orchestration shares nothing with text-to-code retrieval. Archive them;
do not let five days of sunk cost argue for a theme whose evaluation kit never arrived.

The compensating fact: **Theme 1 needs nothing from the organisers.** The dataset, the metric,
the harness and the baselines are all public today. That is why 6 days is enough here and would
not have been there.
