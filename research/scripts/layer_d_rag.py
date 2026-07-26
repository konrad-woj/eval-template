"""Layer D -- RAG-specific metrics.

D9. Retrieval quality: precision@k, recall@k, hit rate@k, MRR, nDCG@k.
D10. Groundedness / faithfulness, context precision@K, context recall.

Distinguishing retrieval quality (D9) from generation quality (faithfulness,
D10) is, per the source research, "the single most useful diagnostic split in
RAG" -- it tells you whether to fix the retriever or the prompt. Stdlib only.
See ../CATALOG.md#layer-d for formulas, sources (RAGAS, TruLens, Evidently,
ir_measures) and failure modes.
"""

from __future__ import annotations

import math

from common import kv, load_fixture, section


def _sorted_by_rank(chunks: list[dict]) -> list[dict]:
    return sorted(chunks, key=lambda c: c["rank"])


def precision_at_k(chunks: list[dict], k: int) -> float:
    top_k = _sorted_by_rank(chunks)[:k]
    return sum(c["is_relevant"] for c in top_k) / len(top_k) if top_k else 0.0


def recall_at_k(chunks: list[dict], k: int) -> float:
    total_relevant = sum(c["is_relevant"] for c in chunks)
    if total_relevant == 0:
        return 0.0
    top_k = _sorted_by_rank(chunks)[:k]
    return sum(c["is_relevant"] for c in top_k) / total_relevant


def hit_rate_at_k(chunks: list[dict], k: int) -> float:
    top_k = _sorted_by_rank(chunks)[:k]
    return 1.0 if any(c["is_relevant"] for c in top_k) else 0.0


def reciprocal_rank(chunks: list[dict]) -> float:
    """MRR for a single query: 1 / rank of first relevant chunk."""
    relevant = [c for c in chunks if c["is_relevant"]]
    if not relevant:
        return 0.0
    return 1.0 / min(c["rank"] for c in relevant)


def _dcg_at_k(chunks: list[dict], k: int) -> float:
    top_k = _sorted_by_rank(chunks)[:k]
    return sum((2 ** c.get("relevance", c["is_relevant"]) - 1) / math.log2(i + 2) for i, c in enumerate(top_k))


def ndcg_at_k(chunks: list[dict], k: int) -> float:
    dcg = _dcg_at_k(chunks, k)
    ideal = sorted(chunks, key=lambda c: -c.get("relevance", c["is_relevant"]))[:k]
    idcg = sum((2 ** c.get("relevance", c["is_relevant"]) - 1) / math.log2(i + 2) for i, c in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# --- D10: groundedness / faithfulness -----------------------------------------


def faithfulness(claims: list[dict]) -> float:
    """RAGAS faithfulness = (# claims supported by context) / (total claims)."""
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c["supported_by_context"])
    return supported / len(claims)


def context_precision_at_k(chunks: list[dict], k: int) -> float:
    """RAGAS context precision@K = sum(precision@k * v_k) / (# relevant in top K)."""
    ranked = _sorted_by_rank(chunks)[:k]
    num_relevant = sum(c["is_relevant"] for c in ranked)
    if num_relevant == 0:
        return 0.0
    total = 0.0
    for i in range(1, len(ranked) + 1):
        v_k = ranked[i - 1]["is_relevant"]
        precision_at_i = sum(c["is_relevant"] for c in ranked[:i]) / i
        total += precision_at_i * v_k
    return total / num_relevant


def context_recall(chunks: list[dict], reference_context_ids: list[str]) -> float:
    """Fraction of reference (ground-truth) context IDs actually retrieved."""
    if not reference_context_ids:
        return 0.0
    retrieved_ids = {c["doc_id"] for c in chunks}
    covered = sum(1 for rid in reference_context_ids if rid in retrieved_ids)
    return covered / len(reference_context_ids)


def run(fixture: dict) -> None:
    chunks = fixture["retrieved_context"]
    k = 3

    section(f"D9: retrieval quality (k={k})")
    kv("precision@k", precision_at_k(chunks, k))
    kv("recall@k", recall_at_k(chunks, k))
    kv("hit_rate@k", hit_rate_at_k(chunks, k))
    kv("MRR (this query)", reciprocal_rank(chunks))
    kv("nDCG@k", ndcg_at_k(chunks, k))

    section("D10: groundedness / faithfulness & context precision/recall")
    kv("faithfulness (claim support ratio)", faithfulness(fixture["answer_claims"]))
    kv(f"context_precision@{k}", context_precision_at_k(chunks, k))
    kv("context_recall", context_recall(chunks, fixture["reference_context_ids"]))


if __name__ == "__main__":
    run(load_fixture())
