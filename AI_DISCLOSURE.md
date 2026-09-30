# AI disclosure

Samsung PRISM GenAI Hackathon 2026 · Theme 1: Agentic Code Intelligence

This project was built by 3 human participants working with AI coding
assistants. Most source code and documentation in this repository was written by
**Claude Code** (Anthropic; models Claude Opus 5, Opus 5.5 and Sonnet 4.6),
working under the participants' direction. The participants chose the theme and
approach, directed each phase, ran every experiment on their own machine,
reviewed the outputs, and made the submission decisions.

Other AI tool used: **[team to fill in: tool name and what it was used for]**

**No generative model runs inside the submitted system.** At runtime it uses one
pretrained *embedding* model, plus a deterministic execution check that uses no model:

| Component | Origin | Licence |
|---|---|---|
| `intfloat/e5-base-v2` (stage-1 encoder, submitted result) | Third-party pretrained model, used unmodified | MIT |
| `Salesforce/SFR-Embedding-Code-400M_R` (optional encoder) | Third-party pretrained model, used unmodified | CC-BY-NC-4.0 |
| CoIR `AppsRetrieval` dataset, MTEB harness | Public benchmark and library | per upstream |

## Per-feature origin

Prompts are quoted where they were short. Most phases were driven by short
instructions such as "resume" or "next phase" against the agreed plan, and are
summarised as such.

| Feature / file | Tool | Prompt (summary) | Output summary | Human review and modifications |
|---|---|---|---|---|
| Theme analysis and plan (`THEME1.md`) | Claude Code | "Make a full analysis of the theme 1 and reverify the plan", with the organisers' theme material | Benchmark analysis, published baselines, risks, six-day plan | Team chose Theme 1 over the earlier Theme 5 work (archived, not submitted). A correction note was added on 26 Sep after full-scale results contradicted one early claim. |
| Dense baseline (`baseline.py`) | Claude Code | Execute plan phase: reproduce E5-base-v2 on AppsRetrieval through MTEB | MTEB runner; reproduces the published 11.52 NDCG@10 exactly | Team ran it locally and checked the score against the published value |
| Execution verifier and sandbox (`execute.py`) | Claude Code | Execute plan phase: run candidate programs on the worked examples in the query, safely | Example parser, isolated `python -I` subprocess runner with timeout and stripped environment, append-only result cache, self-test | Self-test run by the team. Sandbox limits on Windows are documented in the README, not overstated. |
| Two-stage pipeline (`pipeline.py`) | Claude Code | Execute plan phase: dense retrieval + execution rerank, scored end to end by MTEB | MTEB `SearchProtocol` implementation; writes the official results JSON and predictions | Team ran the full 3,765-query benchmark (20.71 NDCG@10). An earlier 30.74 figure from a 150-query sample was withdrawn once the full run was done. |
| CodeXEmbed support (`pipeline.py`) | Claude Code | Run a code-retrieval embedder on CPU with current libraries | Compatibility fixes for transformers 5, fp32 inference, 1,024-token cap | Full run did not complete before the deadline, so no score is claimed |
| Offline analysis (`analyze.py`) | Claude Code | Ablations from cached embeddings and execution results | Recomputes MTEB scores offline; ablations in `RESULTS.md` | Team cross-checked against the MTEB JSON |
| Demo (`search.py`) | Claude Code | Query in, ranked solutions out, with timings | CLI demo showing dense rank, execution result and final rank | Team ran it for the demo video |
| Content-addressed indexes (P1) | Claude Code | Support retrieval across code versions | Embedding store keyed by snippet text; execution cache keyed by program content + examples | Reviewed by the team |
| README, RESULTS, Dockerfile, requirements | Claude Code | Document approach, verified results and setup | Documentation and environment files | Reviewed by the team |
| Final packaging (30 Sep) | Claude Code | "finalise the project and get it ready for submission" | Doc corrections, demo default set to the submitted configuration, presentation deck filled into the organisers' template, this disclosure | Team reviewed and added team details and the demo video link |
