"""Layer H -- Online/production monitoring & statistical rigor.

H17. Operational metrics: p50/p95 latency, mean cost/tokens/steps -- the
     things you'd export to Evidently/Langfuse/Arize Phoenix in production.
H18. Statistical rigor: bootstrap confidence intervals, normal-approximation
     CI (Evan Miller, arXiv 2411.00640), and Expected Calibration Error (ECE).

Stdlib only (`random`, `statistics`, `math` -- no numpy/scipy). See
../CATALOG.md#layer-h for sources and thresholds.
"""

from __future__ import annotations

import math
import random
import statistics

from common import kv, load_fixture, section


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def operational_summary(records: list[dict]) -> dict:
    latencies = sorted(r["latency_ms"] for r in records)
    return {
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in records),
        "mean_tokens": statistics.mean(r["tokens"] for r in records),
        "mean_steps": statistics.mean(r["num_steps"] for r in records),
    }


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 42) -> dict:
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_resamples)]
    means.sort()
    lower_idx = int((1 - ci) / 2 * n_resamples)
    upper_idx = int((1 - (1 - ci) / 2) * n_resamples) - 1
    return {"mean": statistics.mean(values), "ci_low": means[lower_idx], "ci_high": means[upper_idx]}


def normal_ci(values: list[float], z: float = 1.96) -> dict:
    """Evan Miller (arXiv 2411.00640): CI_95 = mean +/- 1.96 * SE."""
    n = len(values)
    mean = statistics.mean(values)
    se = statistics.pstdev(values) / math.sqrt(n) if n > 0 else 0.0
    return {"mean": mean, "ci_low": mean - z * se, "ci_high": mean + z * se}


def expected_calibration_error(samples: list[dict], n_bins: int = 5) -> float:
    """ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|. `samples` is a list of
    {"confidence": float in [0,1], "correct": bool}."""
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for s in samples:
        idx = min(int(s["confidence"] * n_bins), n_bins - 1)
        bins[idx].append(s)
    n = len(samples)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if not b:
            continue
        bin_conf = statistics.mean(s["confidence"] for s in b)
        bin_acc = statistics.mean(1.0 if s["correct"] else 0.0 for s in b)
        ece += (len(b) / n) * abs(bin_acc - bin_conf)
    return ece


def run(fixture: dict) -> None:
    section("H17: operational metrics")
    for label, value in operational_summary(fixture["operational_records"]).items():
        kv(label, value)

    section("H18: statistical rigor")
    latencies = [float(r["latency_ms"]) for r in fixture["operational_records"]]
    kv("bootstrap 95% CI on mean latency_ms", bootstrap_ci(latencies))
    kv("normal-approx 95% CI on mean latency_ms", normal_ci(latencies))
    kv("ECE(calibration_samples)", expected_calibration_error(fixture["calibration_samples"]))


if __name__ == "__main__":
    run(load_fixture())
