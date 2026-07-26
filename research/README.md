# research/ — eval methodology review

This directory is a **reference library, not a package**: a catalog of AI/LLM
evaluation methodologies plus runnable, dependency-free demo scripts that
show each one working end-to-end. It exists so a future coding session (human
or agent) starting a real eval pipeline in this repo has a complete list of
viable options in front of it, instead of reaching for the first metric that
comes to mind.

Nothing in here is meant to be imported by production code — it deliberately
lives outside the app source tree so it doesn't shape or get confused with
whatever real eval packages get introduced later.

## Start here

1. **`CATALOG.md`** — the reference catalog: 18 methodologies across 8
   layers (deterministic checks → semantic metrics → LLM-as-judge → RAG →
   agents → safety → human/meta-eval → production monitoring), each tagged
   STABLE/EMERGING/RESEARCH with its canonical source, formulas, failure
   modes, and production tooling. **Read this before picking an eval
   approach.**
2. **`scripts/`** — one Python module per layer, each implementing that
   layer's metrics from the underlying formulas (stdlib only — no API keys,
   no model downloads). Run them to see real numbers, then read the code as
   a starting point for a production implementation.
3. **`fixtures/shared_fixture.json`** — one JSON fixture that drives every
   script, matching the schema documented in `CATALOG.md`.
4. **`CHANGELOG.md`** — dated log of research refreshes. This review gets
   re-run periodically to extend the catalog as the field moves; that log is
   how you tell what changed between refreshes.

## Quick start

```bash
cd research
uv run scripts/run_all.py           # every layer, one shared fixture
uv run scripts/layer_d_rag.py       # or just one layer
```

`uv run` works out of the box with zero extras installed — the demo scripts
have no dependencies beyond the Python 3 standard library. `python3
scripts/run_all.py` works identically if you'd rather not use uv at all.

## Layout

```
research/
├── README.md              you are here
├── CATALOG.md              the reference catalog (18 methods, 8 layers)
├── CHANGELOG.md            dated log of research refreshes
├── pyproject.toml          uv-managed optional deps for graduating demo -> production
├── fixtures/
│   └── shared_fixture.json one fixture, drives every script
└── scripts/
    ├── common.py                     fixture loader + small helpers
    ├── layer_a_deterministic.py      A1 assertions/pass@k, A2 BLEU/ROUGE-L
    ├── layer_b_semantic.py           B3 similarity, B4 hallucination proxy
    ├── layer_c_llm_judge.py          C5 G-Eval, C6 pairwise, C7 jury, C8 DAG
    ├── layer_d_rag.py                D9 retrieval, D10 faithfulness
    ├── layer_e_agent.py              E11 tool-call, E12 trajectory, E13 goal+pass^k
    ├── layer_f_safety.py             F14 PII/toxicity, F15 prompt-injection
    ├── layer_g_human_meta.py         G16 kappa/alpha/spearman, judge alignment
    ├── layer_h_stats_monitoring.py   H17 operational, H18 CI/ECE
    └── run_all.py                    runs every layer against the fixture
```

## Why the demos are stdlib-only

Every metric is implemented straight from its published formula using only
`math`, `statistics`, `random`, `re`, `json`, `collections` — no
`sentence-transformers`, no `scipy`, no LLM API calls. This keeps the
directory runnable with zero setup and makes the mechanics of each metric
inspectable in ~50 lines instead of hidden behind a library. It is **not** a
claim that these are production-grade: `layer_b_semantic.py`'s bag-of-words
cosine similarity is a deliberately weak stand-in for real embeddings, the
LLM judges in `layer_c_llm_judge.py` are heuristic mocks standing in for a
real judge model, and the PII/toxicity detectors in `layer_f_safety.py` are
toy regex/keyword checks. Each script's docstring says exactly what
production tool it stands in for (RAGAS, DeepEval, Evidently, BFCL,
tau-bench, Presidio, Detoxify, etc.) — see `pyproject.toml` and `CATALOG.md`
for the real thing.

To install one layer's real tooling (managed with **uv**, not pip):

```bash
uv sync --directory research --extra layer-d   # installs ragas, trulens-eval, ...
uv add --directory research --optional layer-d some-other-package
```

`research/uv.lock` is intentionally gitignored — it's a 400+ package
universal lock across every optional layer, regenerated on demand rather than
tracked, so it doesn't bloat the repo for a directory nothing depends on by
default.

## How to extend this catalog

This review is meant to be re-run periodically as the field moves. When
doing a refresh:

1. Research the delta (new papers, new benchmarks, tooling maturity changes,
   deprecated methods) rather than starting over — `CATALOG.md`'s STABLE/
   EMERGING/RESEARCH tags are a mid-2026 snapshot and will drift.
2. Update or add entries in `CATALOG.md` under the relevant layer (add a new
   layer only if a method genuinely doesn't fit the existing 8).
3. If the new method is worth a runnable demo, add `layer_x_*.py` (or extend
   an existing layer script) and register it in `scripts/run_all.py`'s
   `LAYERS` list. Add any new fixture fields to
   `fixtures/shared_fixture.json` — **extend, don't remove/rename** existing
   fields, since older scripts depend on them.
4. Add a dated entry to `CHANGELOG.md` describing what changed and why (new
   source, tooling update, correction).
5. Re-run `uv run --directory research scripts/run_all.py` and confirm it
   still completes cleanly before committing. If the new method needs a real
   dependency, add it to the relevant `layer-x` extra (or a new one) in
   `pyproject.toml` rather than importing it in the stdlib-only demo scripts.

## Referenced from the main README

The top-level `README.md` links here in one short section — this catalog is
deliberately kept out of the main repo narrative so it doesn't dominate or
get mistaken for the actual eval pipeline this template will eventually
ship.
