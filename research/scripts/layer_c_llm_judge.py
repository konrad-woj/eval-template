"""Layer C -- LLM-as-judge family.

C5. G-Eval / criteria-based pointwise judge (Liu et al. 2023).
C6. Pairwise LLM-as-judge (Zheng et al. 2023, MT-Bench/Chatbot Arena).
C7. LLM Juries / panel-of-judges (Verga et al. 2024, PoLL).
C8. Deterministic decision-tree scoring, i.e. DeepEval's "DAG" (Confident AI, 2025).

There is no LLM call in this repo -- these demos use `mock_judge_*` heuristics
(token overlap with the question, deterministic) standing in for a real judge
model. To use for real: replace `Judge` with a function that calls your LLM
provider and, for G-Eval specifically, requests output-token logprobs (or
falls back to sampling+averaging if the API doesn't expose them -- see
../CATALOG.md caveats). Every judge here MUST be meta-evaluated against human
labels before you trust it (see layer_g_human_meta.py).
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable

from common import kv, load_fixture, section, tokenize

Judge = Callable[[str, str], float]  # (question, answer) -> heuristic score in [1, 5]


def mock_judge_a(question: str, answer: str) -> float:
    """Heuristic: reward answers that share more content words with the question."""
    q_tokens, a_tokens = set(tokenize(question)), set(tokenize(answer))
    overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    return 1 + 4 * min(overlap * 1.5, 1.0)


def mock_judge_b(question: str, answer: str) -> float:
    """A second, differently-biased heuristic (simulates a different judge model):
    rewards conciseness relative to the question length."""
    ratio = len(tokenize(answer)) / max(len(tokenize(question)), 1)
    closeness = 1.0 - min(abs(ratio - 3.0) / 6.0, 1.0)
    return 1 + 4 * closeness


def mock_judge_c(question: str, answer: str) -> float:
    """A third heuristic: rewards presence of a specific date/number (proxy for
    "answers the question with a concrete fact")."""
    return 5.0 if re.search(r"\d", answer) else 2.0


# --- C5: G-Eval ---------------------------------------------------------------


def g_eval_score(question: str, answer: str, judge: Judge = mock_judge_a) -> float:
    """G-Eval (Liu et al. 2023): score = sum(p(s_i) * s_i), a probability-weighted
    sum over candidate scores. Real G-Eval reads p(s_i) off the judge LLM's
    output-token logprobs; here we synthesize a peaked distribution around the
    heuristic judge's point estimate to demonstrate the weighting mechanics.
    """
    base = judge(question, answer)
    scores = [1, 2, 3, 4, 5]
    weights = [math.exp(-abs(s - base)) for s in scores]
    total = sum(weights)
    probs = [w / total for w in weights]
    return sum(p * s for p, s in zip(probs, scores, strict=True))


# --- C6: pairwise judge ---------------------------------------------------------


def pairwise_judge(question: str, answer_a: str, answer_b: str, judge: Judge = mock_judge_a) -> str:
    """Zheng et al. 2023: judge picks a winner between two answers."""
    score_a, score_b = judge(question, answer_a), judge(question, answer_b)
    if score_a == score_b:
        return "tie"
    return "a" if score_a > score_b else "b"


# --- C7: LLM juries --------------------------------------------------------------


def jury_score(
    question: str, answer: str, judges: tuple[Judge, ...] = (mock_judge_a, mock_judge_b, mock_judge_c)
) -> dict:
    """Verga et al. 2024 (PoLL): aggregate scores from a panel of diverse judges."""
    scores = [j(question, answer) for j in judges]
    return {"scores": scores, "mean": statistics.mean(scores), "stdev": statistics.pstdev(scores)}


# --- C8: deterministic decision-tree scoring ("DAG") ------------------------------


def dag_score(answer: str, claims: list[dict]) -> dict:
    """Toy DeepEval-style DAG: TaskNode -> JudgementNode -> VerdictNode, fully
    rule-based control flow (no LLM call needed for this particular tree).
    """
    has_date = bool(re.search(r"\b(19|20)\d{2}\b", answer))  # TaskNode
    all_supported = all(c["supported_by_context"] for c in claims)  # JudgementNode

    if not has_date:  # VerdictNode
        return {"score": 0.0, "reason": "no date found in answer"}
    if not all_supported:
        return {"score": 0.5, "reason": "date present but at least one claim is unsupported"}
    return {"score": 1.0, "reason": "date present and all claims are supported"}


def run(fixture: dict) -> None:
    question, answer = fixture["question"], fixture["generated_answer"]

    section("C5: G-Eval (probability-weighted pointwise judge)")
    kv("g_eval_score(question, generated_answer)", g_eval_score(question, answer))

    section("C6: pairwise LLM-as-judge")
    winner = pairwise_judge(question, fixture["generated_answer"], fixture["reference_answer"])
    kv("pairwise_judge(generated vs reference)", winner)

    section("C7: LLM jury / panel-of-judges (PoLL)")
    jury = jury_score(question, answer)
    kv("jury scores", jury["scores"])
    kv("jury mean", jury["mean"])

    section("C8: deterministic decision-tree scoring (DAG)")
    dag = dag_score(answer, fixture["answer_claims"])
    kv("dag_score", dag["score"])
    kv("dag reason", dag["reason"])


if __name__ == "__main__":
    run(load_fixture())
