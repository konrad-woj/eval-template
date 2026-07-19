# eval-template — Plan

A standalone, uv-managed package for evaluating LLMs, RAG pipelines, and agents — single-
node or multi-node (e.g. LangGraph) or multi-agent — against **any compatible endpoint**,
with reporting that compares runs against each other and against published benchmark
numbers.

This document is the working plan. It gets revised as decisions are made — treat it as a
living design doc, not a spec frozen in time.

> Revision history: this version consolidates two prior drafts (an initial architecture
> pass, then a revision cross-checked against external production-eval best-practice
> sources — MLflow, LangChain, Latitude, a 12-metric production-agent-eval study, Inspect
> AI's own docs, AWS Nova eval-in-container docs) plus a scoping conversation that changed
> how RAG and agents relate and how eval granularity is layered. See "Decisions log" for
> the full history.

## Goals

- **Agents are the primary target; RAG is a component, not a separate track.** In real
  systems, agents use RAG as a tool/node, and RAG pipelines can retrieve in an agentic
  (multi-step) way — one framework must cover both, sharing the same core abstractions,
  rather than treating "RAG eval" and "agent eval" as two different tools bolted together.
- Works against **any** system under test that speaks one small HTTP contract — no bespoke
  SDK integration per target repo/framework.
- Reproducible, benchmark-grade results: run a fixed dataset, get a score, diff it against a
  previous run and against literature (BEIR, CRAG, etc.) numbers.
- Reporting that slices by task difficulty/type (simple/hard, verbatim/multi-hop), not just a
  single aggregate number.
- Easy to point at a *different* repo/pipeline without modifying this package — config, not code.
- **Eval granularity is layered, built bottom-up**: single tool/node/LLM-call correctness
  first, then path/route correctness (was the right sequence of nodes/tools taken), then
  whole-episode task completion, then cross-agent handoff correctness last — reflecting
  that multi-agent systems are both rarer in practice and harder to evaluate reliably.
- Built for more than one team to depend on: node/agent graph structure and scorers must
  work generically across frameworks (LangGraph, custom orchestration, plain tool-calling),
  not hardcode one framework's internals into the wire contract.

## Non-goals (for now)

- Not a hosted/collaborative platform (no dashboards-as-a-service, no human annotation UI) —
  that's what Langfuse or similar is for, layered on top later if needed.
- Not a production observability/tracing tool — this is an offline/CI eval harness. Online
  eval (production trace sampling, routing live failures back into eval datasets) is a
  deliberate exclusion, not an oversight — revisit only if this package needs to consume
  traces from a tracing tool later, not reimplement one.
- Not reimplementing an execution engine from scratch — see "Build vs. buy" below.
- Not requiring sandboxed execution for every eval — sandboxing (Inspect's Docker sandboxes)
  is an opt-in, per-suite capability for code-executing agent tasks, not a platform-wide
  requirement. Most tool-calling/RAG evals never need it.

## Build vs. buy

Reuse [Inspect AI](https://inspect.aisi.org.uk/) (UK AISI, MIT) as the execution engine:
async concurrency, retries, resumable runs, transcript logging, and — for suites that need
it — sandboxed agent execution all come for free. What this package actually owns and adds
value with:

1. The **wire contract** (`EvalRequest`/`EvalResponse` with a `steps` trace, generalized to
   a graph of steps — see below) that lets *any* HTTP endpoint be evaluated, whether it's a
   bare LLM, a RAG pipeline, a single tool-calling agent, a multi-node LangGraph-style
   agent, or (eventually) a multi-agent system.
2. **Layered scorers** (unit/tool/node-level, path/route-level, episode-level, and
   multi-agent-level — see "Scorers" below), including RAG-specific ones (retrieval
   metrics, faithfulness/context-precision) that plug in as one *kind* of node, not a
   separate contract.
3. **Reporting/analysis**: slicing, run-vs-run regression detection, run-vs-published-baseline
   comparison, cross-target leaderboards.

Concretely: the generic HTTP target is implemented as a custom Inspect **`ModelAPI`**
provider (not a `Solver` — see "Key architectural decision" below), and `inspect eval` (or
`inspect_ai`'s Python `eval()`) drives execution under the hood. This package is the
contract + scorers + reporting layer on top.

This split is externally validated, not just an internal preference: AWS's Nova
Inspect-in-container eval flow (a YAML recipe → `inference_provider` block → benchmarks →
Inspect `eval()` under the hood) is structurally the same shape as our own
`http_target.py` + `suites/*.yaml` design — same "contract + config in front of Inspect"
pattern, applied to a different provider ecosystem.

## Key architectural decision: HTTP target as `ModelAPI`, not `Solver`

Inspect offers two ways to call an external system from inside an eval: a custom `Solver`
that hand-rolls an HTTP call, or a custom `ModelAPI` provider (subclass `ModelAPI`,
implement `async def generate(...)`, register via `@modelapi`, invoke as `--model
http_target/<config>`). We use **`ModelAPI`**: concurrency (`max_connections`), retry
classification (`should_retry`), usage accounting, and transcript logging all come for
free, and every built-in scorer/solver works unmodified against it. A `Solver` is the
fallback only if a target's wire shape can't be squeezed into "list of chat messages in →
completion out" (not expected to be a real problem given our own `EvalRequest`/`EvalResponse`
contract).

**Open risk (Phase 1 spike required)**: `ModelAPI.generate()`'s signature (`input`, `tools`,
`tool_choice`, `config`) does not receive `TaskState`/`store` directly, unlike a `Solver`.
To get `EvalResponse.steps` (retrieved docs, tool calls, node/agent graph structure)
through to scorers, the intended mechanism is `transcript().info({...}, source="target")`
— a contextvar-scoped call that should work from within `generate()`'s async execution
context, read back by scorers via `state.store`/transcript events. **This must be
validated with a throwaway spike at the start of Phase 1** before retrieval/RAG-quality/
path-level scorers are built on top of it. If it doesn't hold up, fall back to a custom
`Solver` wrapping the HTTP call directly (loses free concurrency/retry/usage accounting,
gains direct `TaskState.store` access).

Cross-checked against Inspect's own docs: `Transcript.info()` creates a JSON-serializable
`InfoEvent`, and `Store` mutations auto-record as `StoreEvent` — both are the documented
mechanism for exactly this kind of custom-data passthrough, and both are readable by
scorers via `state.store`/transcript events. This is external confirmation that the spike
is very likely to resolve favorably — it stays a spike (cheap insurance), not a redesign.

**Target capacity sizing**: size `ModelAPI`'s `max_connections` to the target's actual
backend capacity, not a flat default — e.g. AWS's Nova container guidance ties concurrency
to instance count (~25 connections per backend instance as a starting point). Document
this as guidance in `config.py`'s `TargetConfig`, defaulting conservatively and letting a
suite/CLI override raise it once the target's real capacity is known.

## Model providers — no LiteLLM layer

Inspect ships ~28 native `@modelapi` providers, including `google` (Gemini directly),
`openai`, `anthropic`, `azure`, `bedrock`, `vertex`, `mistral`, `groq`, `together`,
`ollama`, `vllm`, `huggingface`, `transformers`, plus a generic `openai-api/<name>/<model>`
passthrough for any OpenAI-compatible endpoint (covers local Unsloth/vLLM servers — the
same pattern `agent-app` itself already uses). LiteLLM would be a redundant abstraction
layer here — Inspect's `--model <provider>/<name>` string swap already gives the same
"change providers without touching code" property, for every provider currently in scope
(cloud Gemini, local Ollama, local Unsloth/vLLM via OpenAI-compat). **No custom LLM
provider code is needed.**

- Default judge model (for LLM-judge scorers): `google/gemini-2.5-flash`, via `GOOGLE_API_KEY`
  in `.env`.
- Config validation enforces `judge_model != target model` whenever the target under test
  is itself a bare LLM — avoids self-grading bias.
- Local-model targets/judges use `ollama/<model>` or `openai-api/<name>/<model>` with a
  base-url env var, no extra code either way.

## Architecture

```
eval-template/
  pyproject.toml
  .env.example                # GOOGLE_API_KEY, target auth headers, etc.
  src/eval_template/
    contracts/                # pydantic schemas — the wire protocol (see below)
      request.py
      response.py
      capabilities.py
    models/
      http_target.py            # @modelapi — wraps EvalRequest/EvalResponse over HTTP
      providers.md              # notes: use native google/ollama/openai-api/*, no custom LLM providers needed
    datasets/
      loaders/                 # each returns list[inspect_ai.dataset.Sample] via hf_dataset()/record_to_sample
        hotpotqa.py              # Phase 1 — starter dataset (see below)
        beir.py                  # Phase 2 (retriever-only unit tests)
        crag.py                  # Phase 5
        jsonl.py                 # bring-your-own dataset
    scorers/
      unit.py                    # single tool-call/node/LLM-call correctness, latency, cost
      retrieval.py              # NDCG/Recall@k/MRR via pytrec_eval — one kind of unit-level scorer
      generation.py             # EM/F1/ROUGE (Inspect's f1()/match() where sufficient; custom where not)
      llm_judge.py              # faithfulness, answer relevancy, context precision/recall (model_graded_qa-based)
      path.py                    # path/route-level: was the right sequence of nodes/tools taken (Phase 6)
      agent.py                  # episode-level: task completion, step efficiency (Phase 6)
      multi_agent.py            # cross-agent handoff correctness — last priority, thin scope (Phase 7+)
    reporting/
      aggregate.py              # per-slice aggregation over EvalLog samples, bootstrap CIs + significance test
      compare.py                # run-vs-run (regression) and run-vs-baseline (capability) diffing, reads .eval logs directly
      render.py                  # Markdown/HTML report, leaderboard table
    calibration/
      judge_calibration.py      # scores llm_judge.py output against a human-reviewed set, reports agreement rate
      gold_trajectories.py      # validates/loads AI-drafted, human-spot-checked "correct path" examples for path.py
    runner.py                  # thin wrapper over inspect_ai.eval()/eval_retry() + our Task/ModelAPI/dataset
    cli.py                     # `evt run|report|compare|calibrate-judge`
    config.py                  # TargetConfig, SuiteConfig (yaml), judge-model validation, max_connections sizing guidance
  suites/                     # bundled task definitions (dataset + scorer selection + gating policy)
    rag_verbatim_simple.yaml
    rag_multihop_hard.yaml
    agent_tool_use.yaml        # phase 6 — single agent, tool-calling
    agent_langgraph_multinode.yaml  # phase 6 — multi-node graph, path-level scoring
  baselines/                  # versioned published paper/leaderboard numbers
    README.md                  # refresh procedure + JSON shape
    beir_2021.json
    crag_kddcup2024.json
  calibration_sets/           # AI-drafted, human-spot-checked judge calibration + gold-trajectory examples
    rag_faithfulness_calibration.jsonl
    langgraph_gold_trajectories.jsonl
  examples/
    fastapi_target_shim/       # canonical, from-scratch reference implementing /v1/evaluate natively
    agent_app_shim/            # adapter: translates agent-app's stateful /v1/chat into the contract
    langgraph_shim/            # adapter: maps LangGraph checkpoint/graph structure into node_id/agent_id steps
  tests/
```

No `targets/` protocol hierarchy (`base.py`/`http.py`/`openai_compat.py`/`python_local.py`) —
the `ModelAPI` *is* the target adapter: one config-driven `models/http_target.py`. No
`datasets/base.py` custom `Sample` protocol — use `inspect_ai.dataset.Sample` directly,
with `metadata` carrying `gold_doc_ids`/`difficulty`/`hop_count`/`task_type`. No
`runner/run_record.py` or `reporting/store.py` — native Inspect `.eval` logs (see below)
replace both.

## Run storage: native `.eval` logs

Inspect's `.eval` logs (`EvalLog`, read via `inspect_ai.log.read_eval_log()`) already give
resumable, replayable, full-fidelity per-sample transcripts (`messages`, `output`, `scores`,
`metadata`, `store`, full `events` list). Reporting builds directly on these under `runs/` —
**no bespoke parquet/jsonl `RunRecord` format**. `inspect eval-retry`/`eval_retry(...)`
gives resumability for free; crash recovery via Inspect's on-disk sample buffer.

Full-transcript-by-default (not just pass/fail) is a named production-eval best practice
(MLflow) — noting it explicitly here as a design win we already get for free via Inspect's
native logs, not a gap to close.

## The wire contract

One request/response schema covers plain-LLM, RAG, single-agent, multi-node, and
(eventually) multi-agent cases via optional fields — not a different schema per case.
Modeled on OpenAI's chat-completions shape (so wrapping an existing endpoint is a few
lines) plus a `steps` trace that is generalized to carry **graph structure**, which is what
unlocks RAG scoring, path/route scoring, and multi-agent scoring from the same contract.

```python
class EvalRequest(BaseModel):
    id: str
    input: str | list[Message]
    context: dict | None = None        # k, available tools, system prompt override
    mode: Literal["completion", "rag", "agent"] | None = None  # hint only

class RetrievedDoc(BaseModel):
    id: str
    text: str
    score: float | None = None
    metadata: dict = {}

class ToolCall(BaseModel):
    name: str
    arguments: dict
    result: str | None = None
    latency_ms: float | None = None

class Step(BaseModel):
    type: Literal["retrieval", "generation", "tool_call", "reasoning"]
    node_id: str | None = None         # which node/tool in the graph produced this step
    agent_id: str | None = None        # which agent produced this step (single agent: constant/omitted)
    parent_step_id: str | None = None  # what step this followed — reconstructs routing/branching
    retrieved: list[RetrievedDoc] | None = None
    tool_call: ToolCall | None = None
    text: str | None = None

class EvalResponse(BaseModel):
    id: str
    output: str
    steps: list[Step] = []             # full trace: retrieval + tool calls + reasoning, as a graph via node_id/parent_step_id
    usage: dict | None = None           # tokens, cost
    latency_ms: float | None = None
    raw: dict | None = None             # escape hatch for target-specific extras
```

`node_id`/`agent_id`/`parent_step_id` are **framework-agnostic by design**: any target can
populate them however its internal structure works (a LangGraph node name, a plain
function name, an agent name in a multi-agent setup). Framework-specific translation
(e.g. mapping LangGraph's checkpoint/graph structure into this shape) lives entirely in a
shim (`examples/langgraph_shim/`), never in the contract itself — this is what lets one
contract serve LangGraph today and a different orchestration framework tomorrow without a
breaking change.

A target implements `POST /v1/evaluate` (request in, response out) and optionally
`GET /v1/capabilities` → `{"supports_retrieval": bool, "supports_tools": bool,
"supports_step_trace": bool, "requires_sandbox": bool, "streaming": bool}` so the runner
can skip suites/scorers the target can't satisfy instead of scoring garbage, and only
spins up Inspect's Docker sandbox when a suite/target actually needs code execution.

A repo becomes eval-compatible by either exposing this route natively, or by getting a small
shim (see `examples/fastapi_target_shim/`) that translates its native call into this schema.
**The one non-negotiable field for RAG/path scoring: `steps` must include retrieved
context and node/agent identity**, or faithfulness/context-precision and path-level
scoring are impossible.

**Episode-only eval as an acceptable floor**: a target that only returns `output` with no
`steps` at all (no node/tool visibility) is not eval-incompatible — it just only supports
the top layer of the granularity ladder (episode-level: task completion, EM/F1/LLM-judge
on final output). Unit-level and path-level scorers self-skip via the capability check
above rather than erroring. This is deliberately the same graceful-degradation pattern the
`agent-app` example already uses, generalized as the documented floor for *any* target,
not a special case.

**Target incompatibility policy**: never adapt the target repo to match this contract.
Always bridge with a thin shim (`examples/fastapi_target_shim/` pattern) implementing
`/v1/evaluate` in front of the unchanged target. `agent-app` (see below) is the first
example of this — treat it as *one* example target, not representative of what all targets
look like; other targets may be OpenAI-shaped, stateless, or expose full steps natively.

### Example: `agent-app`'s `/v1/chat`

Explored as a first candidate target. Its `/v1/chat` (`app/routers.py:167`,
`app/models.py:12-27`) is stateful (`thread_id` + server-side LangGraph checkpoint
history), single-message-per-call (not an OpenAI messages-list shape), and its
non-streaming response (`final_answer`) carries no retrieval/tool-call trace. Per the
incompatibility policy above, `agent-app` itself is left unchanged; `examples/agent_app_shim/`
bridges it — generating/reusing a `thread_id` per sample, mapping `EvalRequest.input` →
`message`, and mapping `final_answer` → `EvalResponse.output` with `steps=[]`. This is the
canonical **episode-only floor** case: no `steps` means only episode-level scoring
(EM/F1/ROUGE, LLM-judge on `final_answer`) applies — unit/path-level scorers self-skip via
the capability check, until/unless `agent-app` is extended to surface a step trace (at
which point `examples/langgraph_shim/`'s mapping approach becomes directly relevant, since
`agent-app` is itself LangGraph-based under the hood).

## Datasets

`Sample` is `inspect_ai.dataset.Sample` (`input`, `target`, `metadata`, ...) — no custom
protocol. Loaders (HotpotQA, BEIR, CRAG, custom JSONL) build samples via `hf_dataset(...,
sample_fields=record_to_sample)` or `json_dataset()`, with `metadata` carrying
`gold_doc_ids`/`difficulty`/`hop_count`/`task_type`. This is what lets reporting slice by
"hard multi-hop" vs "simple verbatim" generically, without per-dataset report code.

**Starter dataset: HotpotQA** (`hotpotqa/hotpot_qa`, `distractor` config, `validation`
split). Smallest dataset that exercises both short-answer scoring (EM/F1, via `answer`) and
retrieval scoring (recall@k/precision@k, via 2 gold + 8 distractor paragraphs +
`supporting_facts`) in one sample — i.e. it is itself a minimal "RAG as a component" case,
scored purely at the unit level, ahead of exercising path/episode-level agent scoring in
Phase 6. Compared alternatives:

| Dataset | Gold docs? | Verdict |
|---|---|---|
| **HotpotQA** | Yes — gold + distractors + supporting facts | **Starter** |
| SQuAD | Trivial (1 passage, no negatives) | Answer-scoring only |
| Natural Questions (full) | Yes, but 45GB+, heavy preprocessing | Too heavy to iterate |
| NQ (`nq_open`) | No — answers only | Fails dual requirement |
| TriviaQA | Yes, noisy/distant-supervised | Usable but noisier |
| MS MARCO | Ranking judgments, weak answer field | Better for IR than QA |
| BEIR | Retrieval-only, no answers | Not a QA dataset — Phase 2 retriever unit tests |
| CRAG (KDD Cup 2024) | Yes, elaborate, high setup cost | "Graduate" benchmark, Phase 5 |

**Process note (Phase 1 prerequisite, not code)**: before building automated scorers,
manually review 20-50 raw target transcripts against the starter dataset. This surfaces
failure modes (bad retrieval, hallucination shapes, formatting quirks) that shape which
scorers/thresholds actually matter, rather than guessing upfront (LangChain's eval
readiness checklist opens with this step).

## Scorers — layered by granularity

`Scorer` follows Inspect's `@scorer` pattern: `async def score(state: TaskState, target:
Target) -> Score`, grading the full `TaskState` (messages, output, metadata, store — not
just final output), so custom scorers can read retrieval/tool-call/graph-structure steps
stashed via the `transcript().info()` mechanism (see spike above). Self-skip via
capability check when a target can't supply what a scorer needs (see "Episode-only eval as
an acceptable floor" above).

Scorers are built and rolled out bottom-up, matching the priority order agreed for this
package — smaller, cheaper, more reliable checks first:

1. **Unit level** (`scorers/unit.py`, `retrieval.py`, `generation.py`) — single tool
   call/node/LLM call: is this one hop correct, how fast, how expensive. RAG retrieval
   (NDCG/Recall/MRR via `pytrec_eval`) and generation (EM/F1/ROUGE, or LLM-judge for
   open-ended answers) are unit-level scorers under this framing — a RAG pipeline
   embedded inside an agent as a tool is scored by the exact same unit-level retrieval
   scorer as a standalone RAG pipeline, no separate contract needed.
2. **RAG-quality (LLM judge)** — faithfulness, answer relevancy, context precision/recall
   (RAGAS-style prompts, model-graded, default judge `google/gemini-2.5-flash`) — also
   unit-level, scoring a single generation step's grounding in retrieved context.
3. **Path/route level** (`scorers/path.py`, Phase 6) — was the sequence of
   nodes/tools/agents taken the *correct* one (right route through the graph, no
   redundant/skipped hops), using `node_id`/`parent_step_id` to reconstruct the path and
   AI-drafted, human-spot-checked "gold trajectories" (see Calibration below) as the
   reference. This is new relative to earlier drafts of this plan and is what actually
   captures LangGraph-style multi-node routing correctness, rather than treating it as
   just "step-count efficiency."
4. **Episode level** (`scorers/agent.py`, Phase 6) — did the full run complete the task,
   often itself a programmatic checker (e.g. "did tests pass" for SWE-bench-style tasks),
   plus step-count efficiency. This is the floor every target supports, including
   episode-only targets with no step trace at all.
5. **Multi-agent level** (`scorers/multi_agent.py`, Phase 7+) — cross-agent handoff
   correctness, redundant work across agents. Deliberately last and thin in scope,
   reflecting that multi-agent setups are both rarer in practice and harder to evaluate
   reliably than single-agent/multi-node cases.

### LLM-judge calibration and gold trajectories (new)

Two related artifacts, same workflow: **AI-drafted, human-spot-checked**, not hand-written
from scratch and not fully automated either.

- **Judge calibration** (`calibration_sets/rag_faithfulness_calibration.jsonl`): before any
  `llm_judge.py` score is trusted for CI gating, draft 20-100 labeled examples by having a
  strong model produce first-pass labels, then have a human spot-check a sample (e.g.
  ~20%) rather than hand-labeling all of them. `calibration/judge_calibration.py` scores
  the judge against this set and reports agreement rate; target ≥75% agreement before that
  judge's output is used as a gating signal — below that, it stays informational-only.
  `evt calibrate-judge --scorer faithfulness` runs this check, re-run whenever the judge
  model or prompt changes.
- **Gold trajectories** (`calibration_sets/langgraph_gold_trajectories.jsonl`): same
  AI-draft + human-spot-check workflow, used as the reference for path/route-level
  scoring (item 3 above) — "what's the correct sequence of nodes for this input."
- **Ownership**: whoever builds a target's shim (e.g. `examples/langgraph_shim/`)
  contributes that target's gold trajectories, since they understand the target's expected
  behavior — `eval-template` defines the format and validates presence/freshness, it
  doesn't author every target's gold data itself.

## Reporting / analysis

1. Sliced aggregation by `metadata` dimensions (difficulty, hop_count, task_type), reading
   directly from `EvalLog` samples — not one overall number.
2. **Regression eval** (run-vs-run): per-sample diff against the immediately prior run,
   gated on a significance test (Welch's t-test) in addition to raw delta — catches "did
   this change break something," tuned for high sensitivity, run on every CI push.
3. **Capability eval** (run-vs-published-baseline): overlay a run against `baselines/*.json`
   (BEIR, CRAG, ...) — answers "how good is this pipeline, absolutely," tuned for a
   different cadence (e.g. weekly/release, not every push) and different thresholds than
   regression eval. Baseline numbers are refreshed periodically by asking Claude Code to
   look up current leaderboard numbers and update the JSON files by hand — no scraping
   code. Procedure and JSON shape (source URL, `retrieved_date`, `scores` map) documented
   in `baselines/README.md` so refreshes stay consistent across sessions.

   Regression and capability eval are kept as explicitly distinct concepts in `compare.py`
   (separate functions/CLI flags, separate default thresholds) rather than one generic
   "diff two things" mode.
4. **CI gating is per-granularity-layer, and each layer's gate is independent** — a
   unit-level regression (one tool got worse), a path-level regression (routing changed),
   and an episode-level regression (task completion dropped) are each their own gate, none
   masking the others. A single blended pass/fail would let, e.g., a routing regression
   hide behind an episode score that still happens to pass.
5. **New scorers start informational-only, get promoted to blocking once proven** — rather
   than a brand-new scorer (especially path-level and multi-agent-level ones, which are
   newest and least battle-tested) blocking merges before its false-positive rate is
   known. Promotion criterion: stable behavior (low false-positive rate against known-good
   runs) over N consecutive runs — exact N and process TBD when the first path-level
   scorer ships in Phase 6, not decided speculatively now.
6. Bootstrap confidence intervals **and** a significance test (Welch's t-test) per metric —
   bootstrap CIs alone were judged insufficient for gating decisions once LLM-judge scorers
   (inherently noisier/non-deterministic) are involved; the t-test gives an explicit
   pass/fail signal for CI thresholds, the CI gives the uncertainty band for humans reading
   the report.
7. Outputs: local Markdown/HTML report (source of truth, no hosted backend required) +
   machine-readable JSON summary for CI threshold gating, broken out per granularity layer
   per item 4.
8. Leaderboard mode: `evt compare --targets a,b,c --suite ...` runs one suite against multiple
   targets and renders a side-by-side table — first-class, since comparing your own pipeline
   variants against each other is a primary use case.

## CLI

```
evt run             --target http://localhost:8000 --suite suites/rag_multihop_hard.yaml --out runs/pipeline-a.eval
evt report           runs/pipeline-a.eval --by difficulty,hop_count
evt compare          runs/pipeline-a.eval runs/pipeline-b.eval              # regression eval
evt compare          runs/pipeline-a.eval --against baselines/crag_kddcup2024.json  # capability eval
evt calibrate-judge  --scorer faithfulness --calibration-set calibration_sets/rag_faithfulness_calibration.jsonl
```

## Roadmap

- [ ] **Phase 0 — Foundations**: `uv init`, package skeleton, lint/typecheck/test CI,
      license/README, `.env.example`.
- [ ] **Phase 1 — Core loop**: spike `transcript().info()` steps-passthrough (see risk
      above) against a minimal mock target; `contracts/` (including `node_id`/`agent_id`/
      `parent_step_id` on `Step`), `models/http_target.py` as an Inspect `ModelAPI`,
      `runner.py` wrapping `inspect_ai.eval()`, HotpotQA loader, `examples/
      fastapi_target_shim/` (canonical native reference) end-to-end against the mock. Then
      validate against a real endpoint: `examples/agent_app_shim/` bridging `agent-app`'s
      `/v1/chat` — episode-only-floor scoring (EM/F1/LLM-judge on `final_answer`), since
      `agent-app` exposes no `steps`. **Prerequisite process step**: manually review 20-50
      raw transcripts from the mock/real target before finalizing which scorers to build
      in Phase 2/3.
- [ ] **Phase 2 — Retrieval-stage eval (unit level)**: BEIR/MTEB loader,
      `scorers/retrieval.py`, dense-vs-sparse comparison against BEIR baseline numbers.
- [ ] **Phase 3 — RAG generation scorers (unit level)**: `llm_judge.py` (faithfulness,
      relevancy, context precision/recall) using `google/gemini-2.5-flash` as default
      distinct judge model, pluggable via config. Includes
      `calibration/judge_calibration.py` + `calibration_sets/rag_faithfulness_calibration.jsonl`
      (AI-drafted, human-spot-checked) and `evt calibrate-judge` — judge scores are not
      used for CI gating until calibrated to ≥75% human agreement.
- [ ] **Phase 4 — Reporting**: `aggregate.py` (bootstrap CIs + Welch's t-test),
      `compare.py` (regression eval vs. capability eval as distinct modes/thresholds, and
      per-granularity-layer independent gates), `render.py`, CLI `report`/`compare`
      commands, `baselines/README.md` + baseline files for BEIR + CRAG.
- [ ] **Phase 5 — Broader RAG suite coverage**: CRAG, MuSiQue, StratRAG loaders; bundled
      `suites/*.yaml` mapped to simple/hard/verbatim/multi-hop task categories.
- [ ] **Phase 6 — Agent eval extension (path + episode level)**: agent datasets
      (GAIA/SWE-bench-style), `examples/langgraph_shim/` mapping LangGraph's
      checkpoint/graph structure into `node_id`/`agent_id`/`parent_step_id`,
      `scorers/path.py` (route correctness against gold trajectories) and `scorers/
      agent.py` (episode-level task completion, step efficiency), optional sandboxed
      execution (via Inspect's Docker sandboxes, gated on `requires_sandbox` capability,
      not default-on) for code-executing tasks. Path-level scoring is scoped in explicitly
      here — per-step reliability compounds multiplicatively over a trajectory (e.g.
      95%/step over 20 steps ≈ 36% end-to-end success), so episode-only scoring materially
      overstates agent quality; this phase does not defer that to "later."
- [ ] **Phase 7 — Multi-agent eval + hardening**: `scorers/multi_agent.py` (cross-agent
      handoff correctness) — deliberately thin scope, last priority, since multi-agent
      setups are rarer and harder to evaluate reliably. Plus: docs, more `examples/`
      shims, versioned/tagged releases for multi-team consumption (see Distribution in
      Decisions log), packaging/consumer-repo integration guide.

## Verification (once implementation starts)

- Phase 1 spike: write a minimal `@task` + mock FastAPI target returning `steps`, run
  `inspect eval` against it, inspect the resulting `.eval` log's events to confirm the
  `transcript().info()` payload round-trips (including `node_id`/`parent_step_id`) and a
  scorer can read it back from `state.store`.
- `examples/fastapi_target_shim/` + HotpotQA sample subset: `evt run` end-to-end, confirm
  EM/F1 and a retrieval metric both populate in the report.
- `examples/agent_app_shim/` against a locally running `agent-app` instance (`AGENT_API_KEY`
  set): confirm the shim correctly maps one `EvalRequest` per sample to a fresh
  `thread_id` + `/v1/chat` call, `final_answer` flows into `EvalResponse.output`, and
  episode-only-floor scoring works with unit/path scorers cleanly self-skipping.
- `examples/langgraph_shim/` (Phase 6) against a LangGraph-based target exposing node
  transitions: confirm `node_id`/`parent_step_id` correctly reconstruct the executed path,
  and `scorers/path.py` scores it against a gold trajectory.
- `evt report`/`evt compare` against two runs of the mock target with deliberately
  different outputs, confirm regression detection and baseline diffing render correctly,
  including the Welch's t-test significance flag and independent per-layer gate results.
- `evt calibrate-judge` against the faithfulness calibration set: confirm the agreement-rate
  report renders and correctly flags a judge below the 75% threshold as "not gate-ready."

## Decisions log (former "open questions")

- Judge model: Inspect native `google/gemini-2.5-flash` default, distinct from target model
  when target is itself a bare LLM (config-enforced). No LiteLLM layer needed — Inspect's
  ~28 native providers + `openai-api/<name>` passthrough cover cloud (Google/OpenAI/
  Anthropic/Azure/Bedrock) and local (Ollama, Unsloth/vLLM via OpenAI-compat) cases already.
- Baseline numbers: hand-refreshed by asking Claude Code to look them up on the web
  periodically; no scraping code; procedure documented in `baselines/README.md`.
- First target to validate against: `agent-app`'s `/v1/chat`, bridged via
  `examples/agent_app_shim/` (target repo itself unchanged) — treated as one example, not
  representative of all targets.
- Run storage: local, native Inspect `.eval` logs — no pluggable backend needed yet.
- Streaming: non-streaming only, sufficient for eval purposes.
- LLM-judge scores require calibration against a human-reviewed set (≥75% agreement)
  before use in CI gating — added after cross-checking against MLflow's and LangChain's
  production-eval guides, which both flag uncalibrated judges as a common failure mode.
- Regression eval (run-vs-run) and capability eval (run-vs-baseline) are kept as distinct
  concepts with separate thresholds/cadence in `compare.py`, not one generic diff.
- Bootstrap CIs alone are insufficient for CI gating; added Welch's t-test as an explicit
  significance signal per metric, on top of the existing CI reporting.
- **Scope: both RAG and agents in scope, agents are the primary target and RAG is treated
  as a component** (a retriever is just one kind of tool/node an agent can call) — not two
  separate tracks. RAG remains the near-term implementation vehicle (Phases 1-5) because
  it's simpler to validate the contract against, but the contract and scorer layering are
  designed agent-first from the start so Phase 6/7 don't require a rework.
- **Agent shape in scope: single-agent tool use, single-agent multi-node (e.g. LangGraph),
  and multi-agent** — prioritized bottom-up by eval granularity: unit (tool/node/LLM call)
  → path/route → episode → multi-agent, reflecting that multi-agent is both rarer in
  practice and harder to evaluate reliably.
- **`node_id`/`agent_id`/`parent_step_id` are generic, framework-agnostic contract fields**;
  LangGraph-specific mapping lives entirely in `examples/langgraph_shim/`, never in the
  contract — preserves the "never adapt the target, always shim" policy and keeps the
  contract stable if a different orchestration framework needs support later.
- **Episode-only eval is an accepted floor**, not a degraded/second-class mode: a target
  with no step trace at all still gets full episode-level scoring; unit/path scorers
  simply self-skip via the capability check. This is the same pattern `agent-app` already
  uses, now generalized and documented as the baseline contract for any target.
- **Sandboxed execution is opt-in per suite/target** (`requires_sandbox` capability flag),
  not a platform-wide requirement — most tool-calling/RAG evals never need Inspect's Docker
  sandboxes; only code-executing agent tasks (e.g. SWE-bench-style) turn it on.
- **Calibration sets and gold trajectories are AI-drafted, then human-spot-checked** (not
  fully hand-written, not fully automated) — cuts curation cost while keeping a human
  check in the loop. Owned by whoever builds a target's shim (they understand that
  target's expected behavior), not centrally authored by the eval-tool maintainers.
  `eval-template` defines the format and validates presence/freshness.
- **CI gating is per-granularity-layer and each layer blocks independently** (unit, path,
  episode, multi-agent each have their own gate) rather than one blended pass/fail, so a
  regression at one layer can't hide behind a passing score at another. New scorers launch
  informational-only and are promoted to blocking once proven stable over multiple runs —
  exact promotion criteria to be defined when the first Phase 6 scorer ships.
- **Distribution: internal, but multi-team from the start** — not a single-team-only tool.
  Versioned/tagged git releases (not just an unversioned git dependency) so other teams'
  target-repo shims can depend on the contract without it breaking under them. Still no
  PyPI/public distribution — that's a later decision if this ever needs to go outside the
  org, and is explicitly not being designed for yet.
- **(Confirmed, no change)** ModelAPI-not-Solver and the `transcript().info()`/`Store`
  passthrough mechanism are Inspect's own documented extension pattern, cross-checked
  against Inspect's docs — the Phase 1 spike is expected to pass, kept only as cheap
  insurance, not because the design is in doubt.
- **(Confirmed, no change)** OpenEnv+Inspect (HuggingFace) evaluated and ruled out as a
  relevant integration — OpenEnv is an RL training-environment abstraction (reset/step/
  reward for GRPO/SFT loops), not an eval-harness pattern, and doesn't overlap with
  evaluating arbitrary HTTP-served production RAG/agent systems.
- **(Confirmed, no change)** Online/production eval (trace sampling, feedback loops into
  eval datasets) stays out of scope per the existing non-goals — this package's charter is
  offline/CI eval, not observability.