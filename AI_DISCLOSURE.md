# AI disclosure

Samsung PRISM GenAI Hackathon 2026 · Theme 1: Agentic Code Intelligence · Team Error404 (VITV)

## Summary

We chose the problem and the approach, directed the build phase by phase from
13 to 30 September, ran every experiment on our own hardware, and did not accept
a result until it had been checked independently. The code and documentation
were produced with Claude Code (Anthropic; models Claude Opus 5, Opus 5.5 and
Sonnet 4.6) as our AI coding assistant, working under our direction. The
per-feature table at the end records the origin of each part.

Other AI tool used: **[tool name, and what we used it for]**

**No generative model runs inside the submitted system.** At runtime it uses one
pretrained embedding model and a deterministic execution check:

| Component | Origin | Licence |
|---|---|---|
| `intfloat/e5-base-v2` (stage-1 encoder, submitted result) | Third-party pretrained model, used unmodified | MIT |
| `Salesforce/SFR-Embedding-Code-400M_R` (optional encoder) | Third-party pretrained model, used unmodified | CC-BY-NC-4.0 |
| CoIR `AppsRetrieval` dataset, MTEB harness | Public benchmark and library | per upstream |

## Decisions we made, and why

Claude Code handled the implementation. Choosing which problem to solve, what
to believe, and what to claim was our job:

1. **We changed theme mid-hackathon.** We started on Theme 5 (interruptible
   real-time agents), researched it in phases and built an engine with
   tests. On 23 September we took the organisers' Theme 1 material, asked for a
   full analysis, and switched. Theme 1 could be measured on a public benchmark
   the same day, while Theme 5's evaluation kit had not arrived. Dropping ten
   days of our own work was the hardest call of the project. The Theme 5 work is
   archived and not submitted.
2. **We chose execution over a bigger model.** Our largest gain comes from
   *running* candidate programs on the worked examples in each problem, not from
   a larger neural network. That fits the theme's CPU-only constraint: a model
   ~65× smaller than E5-Mistral-7B matches it on this benchmark.
3. **No number counts until it holds on the full test split.** An early run on a
   150-query sample gave 30.74 NDCG@10, which would have beaten the published
   state of the art. We did not claim it. On all 3,765 queries the true figure
   is 20.71, and that is the only number we report.
4. **We verified results independently.** The headline score was checked three
   ways before it went into the deck: the official MTEB JSON, an offline
   recomputation (`analyze.py`), and an independent rescoring of all 3,765 saved
   predictions. We also commissioned a written audit of the project on 27
   September, which found gaps that we then fixed.
5. **We kept negative results.** `RESULTS.md` records what did not work:
   selective boosting, runs where the reranker hurts, and spurious timeouts when
   execution competes with the encoder for CPU.
6. **We submitted what was finished.** On the last day the stronger CodeXEmbed
   run was incomplete. We submitted the verified E5 result instead of a number
   we could not stand behind, and marked CodeXEmbed as future work.

## Per-feature origin

Prompts are quoted where they were short. Most implementation phases were driven
by short instructions such as "resume" or "next phase" against a plan we had
agreed, and are summarised as such.

| Feature / file | Tool | Prompt (summary) | Output summary | Our review and modifications |
|---|---|---|---|---|
| Theme research (Theme 5, archived) | Claude Code | "make a proper research on the theme first, run the research in phases, instead of a single session" | Phased research, an agent engine and tests | Not submitted; superseded by our switch to Theme 1 |
| Theme 1 analysis and plan (`THEME1.md`) | Claude Code | "Make a full analysis of the theme 1 and reverify the plan", with the organisers' theme material | Benchmark analysis, published baselines, risks, six-day plan | We chose Theme 1 on the final form. A correction note was added on 26 Sep after full-scale results contradicted one early claim. |
| Dense baseline (`baseline.py`) | Claude Code | Plan phase: reproduce E5-base-v2 on AppsRetrieval through MTEB | MTEB runner; reproduces the published 11.52 NDCG@10 exactly | Run on our machine; score checked against the published value |
| Execution verifier and sandbox (`execute.py`) | Claude Code | Plan phase: run candidate programs on the worked examples in the query, safely | Example parser, isolated `python -I` subprocess runner with timeout and stripped environment, append-only result cache, self-test | Self-test run. Windows sandbox limits are stated in the README, not overstated. |
| Two-stage pipeline (`pipeline.py`) | Claude Code | Plan phase: dense retrieval + execution rerank, scored end to end by MTEB | MTEB `SearchProtocol` implementation; writes the official results JSON and predictions | Full 3,765-query benchmark run on our machine (20.71 NDCG@10). The 150-query sample figure (30.74) was withdrawn. |
| CodeXEmbed support (`pipeline.py`) | Claude Code | Run a code-retrieval embedder on CPU with current libraries | Compatibility fixes for transformers 5, fp32 inference, 1,024-token cap | Full run not finished before the deadline, so no score is claimed |
| Offline analysis (`analyze.py`) | Claude Code | Ablations from cached embeddings and execution results | Recomputes MTEB scores offline; ablations in `RESULTS.md` | Cross-checked against the MTEB JSON |
| Demo (`search.py`) | Claude Code | Query in, ranked solutions out, with timings | CLI demo showing dense rank, execution result and final rank | Run and recorded by us for the demo video |
| Content-addressed indexes (P1) | Claude Code | Support retrieval across code versions | Embedding store keyed by snippet text; execution cache keyed by program content + examples | Reviewed by us |
| README, RESULTS, Dockerfile, requirements | Claude Code | Document approach, verified results and setup | Documentation and environment files | Reviewed by us |
| Final packaging (30 Sep) | Claude Code | "finalise the project and get it ready for submission" | Doc corrections, demo default set to the submitted configuration, presentation filled into the organisers' template, disclosure draft | Team details, member contributions and demo video added by us |
