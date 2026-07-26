"""Layer G -- Human & meta-evaluation.

G16. Judge alignment / inter-annotator agreement: Cohen's kappa, Krippendorff's
alpha (nominal, handles missing ratings), and Spearman rank correlation --
used to check judge_scores against human_scores before trusting an LLM judge.
Rough go/no-go per the source research: Cohen's kappa >= 0.6 or Spearman
rho >= 0.7; below kappa ~= 0.4, fall back to human eval or redesign the rubric.

Stdlib only (no scipy/sklearn). See ../CATALOG.md#layer-g for sources
(Cohen 1960; Krippendorff 1980/2013; Shankar et al. EvalGen 2024; Hamel Husain
open/axial coding).
"""

from __future__ import annotations

from collections import defaultdict

from common import kv, load_fixture, section


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """kappa = (p_o - p_e) / (1 - p_e)."""
    n = len(labels_a)
    assert n == len(labels_b), "label lists must be parallel over the same items"
    categories = sorted(set(labels_a) | set(labels_b))
    p_o = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    p_e = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def krippendorff_alpha_nominal(reliability_matrix: list[list]) -> float:
    """Krippendorff's alpha, nominal metric, coincidence-matrix formulation.
    `reliability_matrix`: one row per unit (item), one column per rater; use
    None for a missing rating. Units with fewer than 2 ratings are skipped.
    Equals Cohen's kappa in the two-rater, complete-data, nominal case.
    """
    o: dict[tuple, float] = defaultdict(float)
    for unit in reliability_matrix:
        values = [v for v in unit if v is not None]
        m_u = len(values)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(m_u):
                if i != j:
                    o[(values[i], values[j])] += 1.0 / (m_u - 1)

    categories = {c for pair in o for c in pair}
    row_sum = {c: sum(o.get((c, k), 0.0) for k in categories) for c in categories}
    n = sum(row_sum.values())
    if n == 0:
        return 1.0

    d_o = sum(v for (c, k), v in o.items() if c != k) / n
    d_e_numerator = sum(row_sum[c] * row_sum[k] for c in categories for k in categories if c != k)
    d_e = d_e_numerator / (n * (n - 1))
    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def spearman(x: list[float], y: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for t in range(i, j + 1):
                ranks[order[t]] = avg_rank
            i = j + 1
        return ranks

    n = len(x)
    if n < 2:
        return 0.0
    rx, ry = rank(x), rank(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True))
    return 1 - (6 * d2) / (n * (n**2 - 1))


def run(fixture: dict) -> None:
    judge_scores, human_scores = fixture["judge_scores"], fixture["human_scores"]

    section("G16: judge <-> human alignment")
    kv("Cohen's kappa(judge_scores, human_scores)", cohen_kappa(judge_scores, human_scores))
    kv("Spearman rho(judge_scores, human_scores)", spearman(judge_scores, human_scores))
    kv("Krippendorff's alpha(annotator_labels)", krippendorff_alpha_nominal(fixture["annotator_labels"]))

    kappa = cohen_kappa(judge_scores, human_scores)
    verdict = (
        "OK to trust (>= 0.6)"
        if kappa >= 0.6
        else ("borderline (0.4-0.6)" if kappa >= 0.4 else "do not trust judge (< 0.4)")
    )
    kv("go/no-go verdict", verdict)


if __name__ == "__main__":
    run(load_fixture())
