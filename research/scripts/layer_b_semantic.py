"""Layer B -- Semantic & model-based metrics.

B3. Embedding/semantic similarity -- demoed here with a stdlib-only bag-of-words
    cosine similarity. This is a stand-in: production should use sentence
    embeddings (sentence-transformers) or BERTScore (Zhang et al. 2020), which
    both need a model download and are intentionally NOT wired in here to keep
    this directory dependency-free. Swap `cosine_similarity` for a real
    embedding-based version when you have model access.
B4. Hallucination detection, reference-free -- a lexical-consistency proxy for
    SelfCheckGPT (Manakul et al. 2023): sample N stochastic generations and
    measure how much they agree; real implementations use NLI/QA-based
    consistency, not lexical overlap.

See ../CATALOG.md#layer-b for sources, formulas, and tooling.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from common import kv, load_fixture, section, tokenize


def cosine_similarity(text_a: str, text_b: str) -> float:
    va, vb = Counter(tokenize(text_a)), Counter(tokenize(text_b))
    common_terms = set(va) & set(vb)
    dot = sum(va[t] * vb[t] for t in common_terms)
    norm_a = math.sqrt(sum(v * v for v in va.values()))
    norm_b = math.sqrt(sum(v * v for v in vb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def selfcheck_consistency(sampled_answers: list[str]) -> dict[str, float]:
    """Average pairwise similarity across N sampled generations of the same
    question. Low consistency is a hallucination-risk proxy (SelfCheckGPT
    intuition): if the model "knows" the answer, resampling should agree.
    """
    if len(sampled_answers) < 2:
        return {"mean_pairwise_similarity": 1.0, "hallucination_risk_proxy": 0.0}
    sims = [cosine_similarity(a, b) for a, b in combinations(sampled_answers, 2)]
    mean_sim = sum(sims) / len(sims)
    return {"mean_pairwise_similarity": mean_sim, "hallucination_risk_proxy": 1.0 - mean_sim}


def run(fixture: dict) -> None:
    section("B3: semantic similarity (bag-of-words cosine stand-in)")
    sim = cosine_similarity(fixture["generated_answer"], fixture["reference_answer"])
    kv("cosine_similarity(generated, reference)", sim)

    section("B4: reference-free hallucination proxy (SelfCheckGPT-style)")
    result = selfcheck_consistency(fixture["sampled_answers"])
    kv("mean_pairwise_similarity across sampled_answers", result["mean_pairwise_similarity"])
    kv("hallucination_risk_proxy (1 - consistency)", result["hallucination_risk_proxy"])


if __name__ == "__main__":
    run(load_fixture())
