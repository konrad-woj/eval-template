# AI/LLM Evaluation Methodology Catalog

Reference catalog of 18 production-applicable evaluation methodologies for
LLM/agent systems, organized in 8 layers. This is the durable artifact:
**read this before choosing an eval approach for a new feature.** It exists
so an agent (or a person) starting a new coding session has the complete set
of viable options in front of them instead of reaching for the first metric
that comes to mind (usually BLEU/ROUGE, which are the *narrowest* fit for
most modern chat/agent work — see Layer A).

Each entry is tagged **STABLE** / **EMERGING** / **RESEARCH**:
- **STABLE** — validated, widely used, safe default.
- **EMERGING** — validated in papers/production at some shops, tooling still
  consolidating.
- **RESEARCH** — promising, not yet a safe default without local validation.

Every entry names its canonical source, its failure modes, its open-source
tooling, and — where one exists — the demo script in `scripts/` that
implements it against `fixtures/shared_fixture.json`.

> Provenance: this catalog was seeded from a research report titled
> *"A Production-Grade Catalog of AI/LLM Evaluation Methodologies"*
> (2026-07-26; see `CHANGELOG.md` for revision history and how to refresh it).

---

## How to choose

1. **Start with Layer A + G16 (error analysis).** Before wiring any LLM
   judge, read 30-50 real traces, build a failure taxonomy, and write cheap
   deterministic assertions. Highest ROI, zero marginal cost.
2. **For RAG, instrument Layer D first** (retrieval precision/recall/MRR/nDCG
   + faithfulness) to localize whether failures are retrieval or generation
   before touching prompts.
3. **For agents, prefer final-state comparison + pass^k (E13) and tool-call
   AST match (E11)**; use strict trajectory match only where the path is
   policy-constrained, otherwise any-order/precision-recall.
4. **Any LLM judge must be meta-evaluated** against >= 50 human labels before
   trust. Rough go/no-go: Cohen's kappa >= 0.6 or Spearman rho >= 0.7; below
   kappa ~= 0.4, fall back to human eval or redesign the rubric.
5. **Ship no eval number without error bars** (bootstrap CI; paired tests for
   A/B comparisons on shared items).
6. **Wire safety + prompt-injection tests (Layer F) into CI**, mapped to the
   OWASP LLM Top 10; gate high-risk tool calls behind human-in-the-loop
   approval.
7. **In production, monitor drift + implicit feedback + cost/latency
   (Layer H)** and re-run the offline suite after every model-version
   upgrade.

**Thresholds that change the plan:** judge-human agreement below kappa ~= 0.4
→ fall back to human eval or redesign the rubric; pass^8 dropping more than
~2x below pass^1 → do reliability engineering before launch; embedding/output
drift beyond control limits → re-evaluate after a model-version change.

---

## Layer A — Deterministic / reference-based checks
*Demo: `scripts/layer_a_deterministic.py`*

### A1. Programmatic assertions & exact/structural match — STABLE
JSON-schema validity, regex, exact match, code-execution unit tests,
`pass@k = 1 - C(n-c, k) / C(n, k)` (Chen et al. 2021, Codex/HumanEval), where
`c` is the number of correct samples among `n`. Highest ROI, zero marginal
cost, fully deterministic.
**Use when:** outputs have verifiable structure (function calls, JSON, code).
**Don't use for:** open-ended prose quality.
**Tooling:** DeepEval `JsonCorrectness`, RAGAS `ExactMatch`/execution-based
DataCompy, pytest, `jsonschema`.

### A2. Reference-based text metrics (BLEU/ROUGE/BERTScore) — STABLE (narrow)
See #6/#7 below for BLEU/BERTScore details.
**ROUGE** (Lin 2004): ROUGE-N = recall of reference n-grams; ROUGE-L uses
Longest Common Subsequence: `R_lcs = LCS(c,r)/|r|`, `P_lcs = LCS(c,r)/|c|`,
`F_lcs = (1+b^2)*R_lcs*P_lcs / (R_lcs + b^2*P_lcs)`.
Source: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," ACL
Text Summarization Branches Out 2004 (aclanthology.org/W04-1013).
**Failure modes:** poor correlation with human judgment on open-ended text;
rewards surface overlap; blind to paraphrase and factuality.
**Use only for:** summarization/translation/extraction with references. Not a
primary quality gate for chat/agents.
**Tooling:** Hugging Face `evaluate`, `rouge-score`, RAGAS `RougeScore`.

---

## Layer B — Semantic & model-based metrics
*Demo: `scripts/layer_b_semantic.py`*

### B3. Embedding/semantic similarity — STABLE
Cosine similarity of sentence embeddings; BERTScore (see #7); RAGAS
`SemanticSimilarity`. Reference-based, cheap, paraphrase-robust, no
factuality signal.
**Tooling:** `sentence-transformers`, `bert-score`, RAGAS.

### B4. Hallucination detection, reference-free — EMERGING
**SelfCheckGPT** (Manakul et al. 2023, EMNLP, arXiv 2303.08896): sample N
stochastic responses, measure consistency (NLI/QA/n-gram/BERTScore variants);
high inconsistency implies likely hallucination.
**FActScore** (Min et al. 2023, arXiv 2305.14251): decompose into atomic
facts, compute % supported by a knowledge source.
**Tooling:** `selfcheckgpt`, RAGAS faithfulness.

---

## Layer C — LLM-as-judge family
*Demo: `scripts/layer_c_llm_judge.py`*

All Layer C methods require judge meta-evaluation (Layer G) before you trust
their numbers.

### C5. G-Eval / criteria-based pointwise judge — STABLE
Quality against custom criteria using an LLM with chain-of-thought evaluation
steps (form-filling paradigm). `score = sum(p(s_i) * s_i)` — probability-
weighted sum over candidate integer scores using the judge's output-token
probabilities, to avoid ties and give continuous scores.
Source: Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human
Alignment," EMNLP 2023 (arXiv 2303.16634). Reports Spearman correlation 0.514
with human judgment on summarization, 0.588 on dialogue generation.
**Failure modes:** needs logprobs for the weighting (many APIs don't expose
them — fall back to sampling multiple scores and averaging, an approximation
of the original formula); inherits LLM-judge biases; sensitive to criteria
wording.
**Tooling:** DeepEval `GEval` (the reference implementation practitioners use).

### C6. Pairwise LLM-as-judge — STABLE
Given two outputs, a judge picks a winner against a rubric; tally win-rate.
Source: Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena," NeurIPS 2023 (arXiv 2306.05685): "strong LLM judges like GPT-4 can
match both controlled and crowdsourced human preferences well, achieving over
80% agreement, the same level of agreement between humans."
**Failure modes (documented in the paper):** position bias (few-shot
prompting improved GPT-4's position-bias consistency from 65.0% zero-shot to
77.5%; swapping order is the common mitigation), verbosity bias,
self-enhancement/self-preference bias, limited reasoning/math ability.
Pairwise is more reliable than pointwise but O(n^2).
**Tooling:** DeepEval, promptfoo, OpenAI simple-evals, LangSmith.

### C7. LLM Juries / Panel — STABLE
Aggregate (average/vote) scores from multiple diverse smaller judges instead
of one large judge.
Source: Verga et al., "Replacing Judges with Juries: Evaluating LLM
Generations with a Panel of Diverse Models," arXiv 2404.18796. PoLL
"outperforms a single large judge, exhibits less intra-model bias due to its
composition of disjoint model families, and does so while being over seven
times less expensive" (panel: Claude Haiku, GPT-3.5-turbo, Command-R).
**Failure modes:** correlated errors if judges share a family; higher
aggregate cost/latency than a single small judge.

### C8. Deterministic decision-tree scoring ("DAG") — STABLE (product term)
DeepEval's DAGMetric: a directed acyclic graph of TaskNodes (transform the
test case), judgement nodes (LLM-as-judge branch decisions), and
verdict/leaf nodes (assign scores). Fully deterministic control flow;
shipped Feb 6, 2025 by Confident AI. **Not a general academic method** — a
productized form of "programmatic rubric / decision-tree scoring."
Source: deepeval.com/docs/metrics-dag.
**Failure modes:** you must author the tree; branch decisions still use an
LLM judge unless purely rule-based.

*(Multi-turn/conversational eval — role adherence, knowledge retention,
coherence, goal completion across turns — folds into C7/E13 rather than
standing alone. Tooling: DeepEval `ConversationalGEval`/
`ConversationalDAGMetric`, RAGAS topic-adherence, tau-bench for the rigorous
multi-turn agent variant.)*

---

## Layer D — RAG-specific
*Demo: `scripts/layer_d_rag.py`*

### D9. Retrieval quality metrics — STABLE
`Precision@k`, `Recall@k`, `HitRate@k = (1/N) * sum(1[rank_i <= k])`,
`MRR = (1/Q) * sum(1/rank_first_relevant)`,
`nDCG@k = DCG@k / IDCG@k` with `DCG@k = sum((2^rel_i - 1) / log2(i+1))`.
**Distinguish retrieval eval from generation eval — this is the single most
useful diagnostic split in RAG.**
**Tooling:** Evidently (15+ ranking metrics), RAGAS, `ir_measures`.

### D10. Groundedness / faithfulness & answer relevance — STABLE
RAGAS faithfulness = (# claims supported by context) / (total claims in
answer); context precision@K = `sum(Precision@k * v_k) / (# relevant in top
K)`, `v_k` in {0,1}; context recall = (# ground-truth claims attributable to
context) / (total GT claims). TruLens RAG Triad (context relevance,
groundedness, answer relevance), 0-3 LLM-judge scale.
**Tooling:** RAGAS, TruLens, DeepEval.

---

## Layer E — Agent-specific
*Demo: `scripts/layer_e_agent.py`*

### E11. Tool/function-call correctness — STABLE
Berkeley Function Calling Leaderboard (BFCL, Gorilla/UC Berkeley, first
released Feb 2024): AST accuracy (function name + argument names/types/values
match, order-insensitive, no execution), executable accuracy (run in
sandbox, compare output), irrelevance detection (correctly emit no call),
relevance detection (emit >= 1 correct call). AST match is a validated,
cheap proxy for execution.
**Tooling:** BFCL harness, RAGAS `ToolCallAccuracy`/`ToolCallF1`.

### E12. Trajectory evaluation — EMERGING
LangChain AgentEvals taxonomy: strict match (identical messages + tool calls
in order, content may differ), in-order match (required tools in correct
relative order, extra harmless calls allowed), any-order match,
precision/recall over tool calls, plus an LLM-as-judge trajectory evaluator.
Deterministic modes need no LLM calls — fast and cheap.
**Failure modes:** brittle when multiple valid paths exist — prefer
any-order/precision-recall or final-state checks in that case.
**Tooling:** LangChain `agentevals` (github.com/langchain-ai/agentevals).

### E13. Goal completion & reliability (final-state + pass^k) — EMERGING
tau-bench (Yao et al. 2024, arXiv 2406.12045): compares database/final state
after the conversation to the annotated goal state; `pass^k` = fraction of
tasks that succeed on all k independent trials (`pass^k = p^k` under
independence, decays exponentially). Original paper: GPT-4o scored 60.4%
retail pass^1 (Claude 3.5 Sonnet 69.2%), while the best function-calling
agent's pass^8 collapsed to roughly 25% in retail — the consistency gap
pass^k is designed to expose.
**Tooling:** tau-bench / tau2-bench (github.com/sierra-research/tau2-bench),
RAGAS `AgentGoalAccuracy`.

---

## Layer F — Safety / security / adversarial
*Demo: `scripts/layer_f_safety.py`*

### F14. Safety classifiers — STABLE
Runs outputs through bias/toxicity/PII classifiers in parallel and flags
violations.
**Tooling:** Detoxify (github.com/unitaryai/detoxify — fine-tuned BERT for
toxicity/severe-toxicity/obscenity/threat/insult/identity-attack), Microsoft
Presidio (github.com/microsoft/presidio — PII detection/anonymization via
NER+regex+checksums; maintainers caveat "there is no guarantee that Presidio
will find all sensitive information"), Llama Guard, NeMo Guardrails,
Perspective API.

### F15. Adversarial & prompt-injection / red-teaming — EMERGING
Direct + indirect prompt injection, jailbreaks, data exfiltration via tool
use, excessive agency. Map to OWASP Top 10 for LLM Applications 2025 (LLM01
Prompt Injection is #1; LLM06 Excessive Agency; genai.owasp.org).
**Tooling:** promptfoo red-team, Microsoft PyRIT, garak, Giskard; benchmarks
AgentDojo, AgentHarm, InjecAgent, ToolEmu. Governance mapping via NIST AI RMF
Generative AI Profile (NIST AI 600-1) and MITRE ATLAS.

---

## Layer G — Human & meta-evaluation
*Demo: `scripts/layer_g_human_meta.py`*

### G16. Error analysis + judge alignment — STABLE (practice)
Hamel Husain's open coding → axial coding workflow (hamel.dev/blog/posts/evals):
read 30-50 real traces, write freeform failure notes ("open coding"), cluster
into failure modes ("axial coding"), write an eval/assertion per failure
mode: "unsuccessful products almost always share a common root cause: a
failure to create robust evaluation systems." Shreya Shankar et al. "Who
Validates the Validators?" / EvalGen (arXiv 2404.12272) aligns LLM judges to
human grades and warns of "criteria drift: users need criteria to grade
outputs, but grading outputs helps users define criteria" — judge criteria
must be co-developed with data, not fixed a priori (see also SPADE, arXiv
2401.03038).
**Key formulas:** Cohen's kappa = `(p_o - p_e) / (1 - p_e)`; Krippendorff's
alpha = `1 - D_o/D_e` (handles missing data, >2 raters, any measurement
level; equals kappa for the nominal two-annotator complete case); quadratic
weighted kappa for ordinal scales. Spearman rho / Kendall tau for
correlation with human labels.
Sources: Cohen 1960; Krippendorff 1980/2013; Landis & Koch 1977
interpretation bands (slight/fair/moderate/substantial/perfect).
**Tooling:** EvalGen, DeepEval, LangSmith annotation queues, `scikit-learn`
(`cohen_kappa_score`), `krippendorff` package, Label Studio.

---

## Layer H — Online / production monitoring & statistical rigor
*Demo: `scripts/layer_h_stats_monitoring.py`*

### H17. Online monitoring, drift & operational metrics — STABLE
Implicit feedback (thumbs, regeneration rate, conversation abandonment, task
success), A/B testing and interleaving, canary/shadow deploys; drift
detection (input/output distribution drift, embedding drift). Cost/latency
as first-class dimensions: p50/p95 latency, cost per resolved task, tokens
per task, steps-to-completion, loop/oscillation detection.
**Tooling:** Evidently (open-source, 100+ built-in metrics incl. embedding
drift + text descriptors for length/sentiment/toxicity/regex), Langfuse,
Arize Phoenix, OpenTelemetry GenAI semantic conventions as tracing substrate.

### H18. Statistical rigor — STABLE (underused)
Confidence intervals on eval scores, bootstrap resampling, paired difference
tests, clustered standard errors for question groups, minimum sample sizes,
variance across seeds.
Canonical: Evan Miller, "Adding Error Bars to Evals," arXiv 2411.00640
(`CI_95 = mean +/- 1.96*SE`; clustered SEs when questions share a
passage/topic; paired tests when comparing two models on the same items).
Calibration: `ECE = sum((|B_m|/N) * |acc(B_m) - conf(B_m)|)` over M
confidence bins.
**Tooling:** `scipy`, `numpy`, `statsmodels`.

---

## Reference-only entries (validated but subsumed above)

These are the two remaining methods from the original 11-method starting
taxonomy, kept here for completeness since they're still valid narrow tools,
but they don't get their own demo script (BLEU/ROUGE cover the same ground
as A2 with a working implementation; human eval underlies all of Layer G).

### BLEU (Papineni et al. 2002) — STABLE (translation/narrow)
`p_n = sum(count_clip(n-gram)) / sum(count(n-gram))`,
`count_clip = min(count, max_ref_count)`; `BP = 1 if c>r else exp(1-r/c)`;
`BLEU = BP * exp(sum(w_n * log p_n))`, typically N=4, `w_n=1/4`.
Source: Papineni, Roukos, Ward, Zhu, ACL 2002.
**Failure modes:** no recall term (BP is a proxy); poor on open-ended
generation; needs smoothing at sentence level. Use SacreBLEU for
standardized tokenization in production.
**Tooling:** SacreBLEU, NLTK, Hugging Face `evaluate`. (Implemented from the
raw formula in `scripts/layer_a_deterministic.py` for zero-dependency demo
purposes — swap in SacreBLEU for anything real.)

### BERTScore (Zhang et al. 2020) — STABLE (semantic similarity)
Token-level semantic similarity via contextual embeddings + greedy matching:
for each candidate token, max cosine similarity to any reference token,
averaged, = precision; reverse direction = recall; F1 = harmonic mean.
Optional IDF weighting; optional baseline rescaling.
Source: Zhang, Kishore, Wu, Weinberger, Artzi, ICLR 2020 (arXiv 1904.09675).
**Failure modes:** no factuality signal; sensitive to model choice, layer,
IDF corpus; domain-dependent correlation.
**Tooling:** `bert-score`, Hugging Face `evaluate`. (Not implemented as a
demo script — requires a real embedding model; `layer_b_semantic.py` uses a
stdlib bag-of-words cosine similarity as a dependency-free stand-in and says
so in its docstring.)

### Human Evaluation — STABLE (gold standard)
Human ratings against defined criteria, with inter-annotator agreement
checked and scores aggregated. See Layer G for the formulas and tooling —
human eval *is* the meta-evaluation layer, not a separate method.

---

## Framework note: UK AISI Inspect (STABLE, open-source)

Beyond the tool-specific frameworks named above, the UK AI Security
Institute's **Inspect** (inspect.aisi.org.uk;
github.com/UKGovernmentBEIS/inspect_ai; MIT license, `pip install
inspect-ai`) is a framework-agnostic backbone worth evaluating separately: an
eval is a **Task** binding a **Dataset** (labelled input/target samples), a
**Solver** (from a single generate call to a full tool-using agent), and a
**Scorer** (from exact-match to model-graded). First-class agent and
sandboxed-execution support, a companion library of 100+ pre-built evals
(`inspect_evals`), used by METR, Apollo Research, and other AI Safety/Security
Institutes. Not demoed here (it's a full framework, not a metric) — evaluate
it directly if/when this template needs a real eval-running harness rather
than standalone metric functions.

---

## Recommended shared fixture

All 18 methods are driven by one JSON fixture
(`fixtures/shared_fixture.json`). Required fields and which methods they
feed:

| Field | Feeds |
|---|---|
| `question`, `reference_answer` | BLEU/ROUGE/BERTScore, semantic similarity, G-Eval |
| `generated_answer` | all output-scoring methods |
| `sampled_answers` | SelfCheckGPT-style consistency (B4) |
| `retrieved_context` (`text`, `doc_id`, `is_relevant`, `relevance`, `rank`) | retrieval metrics, context precision/recall |
| `reference_context_ids` | context recall |
| `answer_claims` (`claim`, `supported_by_context`) | faithfulness/FActScore, DAG |
| `trajectory` (`role`, `tool_name`, `tool_args`, `tool_output`) | trajectory match, tool-call AST match |
| `expected_trajectory`, `expected_calls` | trajectory/AST match |
| `expected_final_state`, `actual_final_state` | tau-bench final-state comparison |
| `pass_at_k_samples`, `pass_hat_k_tasks` | pass@k, pass^k |
| `multi_turn` | conversational eval |
| `judge_scores`, `human_scores`, `annotator_labels` | judge alignment, kappa/alpha, calibration |
| `calibration_samples` | ECE |
| `latency_ms`, `cost_usd`, `tokens`, `num_steps`, `operational_records` | operational metrics, bootstrap/normal CI |
| `safety`, `safety_test_inputs` | safety + prompt-injection layer |

## Caveats

- Tooling-maturity tags reflect mid-2026; the LLM-judge and agent-eval spaces
  move fast — re-check EMERGING/RESEARCH tags on each refresh (see
  `CHANGELOG.md`).
- "DAG" (Confident AI/DeepEval) and "RAG Triad" (TruLens) are vendor terms —
  the underlying ideas are general and re-implementable, which is what the
  demo scripts do.
- Some formulas (RAGAS context precision, G-Eval token weighting) have
  implementation variants across library versions; verify against the
  pinned version you build against before treating a demo script's output as
  ground truth.
- Benchmark scores (tau-bench, BFCL) age quickly and are model-specific —
  treat the *metric* as durable and the *leaderboard number* as perishable;
  watch for benchmark contamination/overfitting to eval sets.
- The G-Eval probability-weighting step requires output-token logprobs; on
  APIs that don't expose them, fall back to sampling multiple scores and
  averaging (an approximation, not the original formula).
- Every demo script in `scripts/` is intentionally stdlib-only and simplified
  — they exist to make the *shape* of each metric runnable and inspectable
  without API keys or heavy dependencies, not to replace RAGAS/DeepEval/
  Evidently/BFCL/tau-bench in a real pipeline. See `pyproject.toml`
  (uv-managed optional dependency groups, one per layer) for what to install
  when you're ready to graduate from demo to production tooling.
