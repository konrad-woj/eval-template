"""Layer A -- Deterministic / reference-based checks.

A1. Programmatic assertions & pass@k (exact/regex/schema match; pass@k formula).
A2. Reference-based text metrics: BLEU (Papineni et al. 2002) and ROUGE-L (Lin 2004),
    implemented from the formulas directly -- no nltk/rouge-score dependency.

Stdlib only. See ../CATALOG.md#layer-a for sources, failure modes, and the
production-grade tooling (SacreBLEU, Hugging Face `evaluate`, `rouge-score`)
these demos stand in for.
"""

from __future__ import annotations

import math
from collections import Counter

from common import kv, load_fixture, ngrams, section, tokenize

# --- A1: programmatic assertions -------------------------------------------------


def exact_match(candidate: str, reference: str) -> bool:
    return candidate.strip() == reference.strip()


def regex_match(text: str, pattern: str) -> bool:
    import re

    return re.search(pattern, text) is not None


def check_schema(obj: dict, schema: dict[str, type]) -> list[str]:
    """Minimal structural check: schema maps field name -> expected python type."""
    errors = []
    for field, expected_type in schema.items():
        if field not in obj:
            errors.append(f"missing field: {field}")
        elif not isinstance(obj[field], expected_type):
            errors.append(f"field {field!r} expected {expected_type}, got {type(obj[field])}")
    return errors


def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k = 1 - C(n-c, k) / C(n, k). Chen et al. 2021 (Codex/HumanEval)."""
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


# --- A2: BLEU / ROUGE-L -----------------------------------------------------------


def bleu_score(candidate: str, references: list[str], max_n: int = 4) -> float:
    """Sentence-level BLEU (Papineni et al. 2002): BP * exp(sum(w_n * log p_n))."""
    cand_tokens = tokenize(candidate)
    ref_token_lists = [tokenize(r) for r in references]
    weights = [1.0 / max_n] * max_n

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(ngrams(cand_tokens, n))
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        max_ref_counts: Counter = Counter()
        for ref_tokens in ref_token_lists:
            for ng, cnt in Counter(ngrams(ref_tokens, n)).items():
                max_ref_counts[ng] = max(max_ref_counts[ng], cnt)
        clipped = sum(min(cnt, max_ref_counts.get(ng, 0)) for ng, cnt in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(clipped / total if total else 0.0)

    geo_mean = (
        0.0
        if min(precisions) == 0
        else math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions, strict=True)))
    )

    c = len(cand_tokens)
    if c == 0:
        return 0.0
    r_tokens = min(ref_token_lists, key=lambda rt: abs(len(rt) - c))
    r = len(r_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c)
    return bp * geo_mean


def _lcs_length(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidate: str, reference: str, beta: float = 1.0) -> dict[str, float]:
    """ROUGE-L (Lin 2004): F_lcs = (1+b^2)*R*P / (R + b^2*P)."""
    c_tokens, r_tokens = tokenize(candidate), tokenize(reference)
    if not c_tokens or not r_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(c_tokens, r_tokens)
    r_lcs = lcs / len(r_tokens)
    p_lcs = lcs / len(c_tokens)
    f_lcs = (1 + beta**2) * r_lcs * p_lcs / (r_lcs + beta**2 * p_lcs) if (r_lcs + p_lcs) > 0 else 0.0
    return {"precision": p_lcs, "recall": r_lcs, "f1": f_lcs}


def run(fixture: dict) -> None:
    section("A1: programmatic assertions & pass@k")
    kv("exact_match(generated, reference)", exact_match(fixture["generated_answer"], fixture["reference_answer"]))
    kv("regex_match(generated, r'30 days?')", regex_match(fixture["generated_answer"], r"30 days?"))
    schema_errors = check_schema(fixture["safety"], {"toxicity": float, "pii_spans": list, "injection_flag": bool})
    kv("schema check on fixture['safety']", "OK" if not schema_errors else schema_errors)
    samples = fixture["pass_at_k_samples"]
    kv(f"pass@{samples['k']} (n={samples['n']}, c={samples['c']})", pass_at_k(samples["n"], samples["c"], samples["k"]))

    section("A2: BLEU / ROUGE-L (reference-based, narrow -- see CATALOG.md)")
    bleu = bleu_score(fixture["generated_answer"], [fixture["reference_answer"]])
    kv("BLEU(generated, reference)", bleu)
    rouge = rouge_l(fixture["generated_answer"], fixture["reference_answer"])
    kv("ROUGE-L F1(generated, reference)", rouge["f1"])
    kv("ROUGE-L precision/recall", f"{rouge['precision']:.4f} / {rouge['recall']:.4f}")


if __name__ == "__main__":
    run(load_fixture())
