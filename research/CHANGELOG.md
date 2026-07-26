# research/ changelog

Dated log of eval-methodology research refreshes. This directory gets
re-reviewed periodically as the field moves; each refresh should add an
entry here describing what changed, so future sessions can tell what's new
without diffing the whole catalog.

Entry format:

```
## YYYY-MM-DD — <short title>
**Source:** <report/paper/announcement that triggered the update>
**Changed:** <what was added/updated/demoted/removed in CATALOG.md and scripts/>
```

---

## 2026-07-26 — Initial catalog (18 methods, 8 layers)
**Source:** "A Production-Grade Catalog of AI/LLM Evaluation Methodologies:
An 18-Method..." research report (uploaded 2026-07-26), which itself audited
and extended an earlier "11 methods" infographic (DailyDoseofDS).

**Changed:**
- Established `research/CATALOG.md` with the full 8-layer / 18-method
  catalog: A (deterministic checks), B (semantic/hallucination), C
  (LLM-as-judge family), D (RAG-specific), E (agent-specific), F
  (safety/adversarial), G (human & meta-evaluation), H (online monitoring &
  statistical rigor).
- Validated but narrowed BLEU/ROUGE/BERTScore to reference-based
  summarization/translation/extraction use cases (not primary quality gates
  for chat/agents).
- Generalized DeepEval's vendor-specific "DAG" term to "deterministic
  rubric / decision-tree scoring."
- Added the methods the original 11-method list omitted: retrieval quality
  metrics, groundedness/faithfulness (RAG triad), tool-call correctness
  (BFCL), goal completion + pass^k (tau-bench), programmatic assertions +
  pass@k, reference-free hallucination detection (SelfCheckGPT/FActScore),
  adversarial/prompt-injection testing (OWASP LLM Top 10), error analysis +
  judge alignment (Hamel Husain, Shankar EvalGen), online/drift monitoring
  (Evidently), and statistical rigor (Evan Miller error bars, ECE).
- Built `research/scripts/` — one stdlib-only Python module per layer
  (`layer_a_deterministic.py` ... `layer_h_stats_monitoring.py`) plus
  `run_all.py`, all driven by `fixtures/shared_fixture.json`.
- Noted UK AISI Inspect as a framework-agnostic eval-running backbone worth
  evaluating separately (not demoed as a metric — it's a harness, not a
  score).
- Replaced the initial `requirements.txt` optional-deps list with a
  uv-managed `pyproject.toml` (`[project.optional-dependencies]`, one
  `layer-x` extra per layer, `[tool.uv] package = false`). Verified
  `uv run scripts/run_all.py` and `uv sync --extra <layer>` both resolve
  cleanly; `research/uv.lock` is gitignored (400+ transitive packages across
  all extras — regenerate on demand, don't track it).

**Open items for the next refresh:**
- BFCL and tau-bench leaderboard numbers cited in `CATALOG.md` are from the
  original papers (BFCL Feb 2024, tau-bench arXiv 2406.12045) and will be
  stale — re-check current leaderboard state and note any newer
  successor benchmarks.
- G-Eval's logprob-weighting caveat (fallback to sample-and-average when an
  API doesn't expose logprobs) should be re-verified against whatever LLM
  provider this repo ends up using.
- No demo script exists yet for UK AISI Inspect or a real embedding-based
  BERTScore/sentence-transformers similarity — intentionally deferred until
  the repo has an actual model/API dependency to justify pulling them in.
