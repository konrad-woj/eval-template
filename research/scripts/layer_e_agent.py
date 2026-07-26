"""Layer E -- Agent-specific metrics.

E11. Tool/function-call correctness -- simplified AST-style match (BFCL).
E12. Trajectory evaluation -- strict / in-order / any-order / precision-recall
     match variants (LangChain AgentEvals taxonomy).
E13. Goal completion & reliability -- final-state comparison (tau-bench) and
     pass^k across repeated trials of the same task.

Stdlib only. See ../CATALOG.md#layer-e for sources and failure modes (strict
match is brittle when multiple valid tool orderings exist -- prefer any-order
or final-state comparison in that case).
"""

from __future__ import annotations

from common import kv, load_fixture, section


def tool_sequence(trajectory: list[dict]) -> list[str]:
    return [step["tool_name"] for step in trajectory if step.get("tool_name")]


# --- E12: trajectory match variants ---------------------------------------------


def strict_match(trajectory: list[dict], expected: list[str]) -> bool:
    return tool_sequence(trajectory) == expected


def in_order_match(trajectory: list[dict], expected: list[str]) -> bool:
    """Required tools appear in relative order; extra calls in between are OK."""
    it = iter(tool_sequence(trajectory))
    return all(tool in it for tool in expected)


def any_order_match(trajectory: list[dict], expected: list[str]) -> bool:
    return set(expected).issubset(set(tool_sequence(trajectory)))


def tool_precision_recall(trajectory: list[dict], expected: list[str]) -> dict[str, float]:
    seq_set, exp_set = set(tool_sequence(trajectory)), set(expected)
    tp = len(seq_set & exp_set)
    precision = tp / len(seq_set) if seq_set else 0.0
    recall = tp / len(exp_set) if exp_set else 0.0
    return {"precision": precision, "recall": recall}


# --- E11: tool-call AST-style match -----------------------------------------------


def ast_match(trajectory: list[dict], expected_calls: list[dict]) -> bool:
    """BFCL-style AST accuracy: function names match, in order, and every
    expected argument key is present (order-insensitive on argument keys,
    no execution). This is a simplified stand-in for the real BFCL harness.
    """
    calls = [
        {"tool_name": s["tool_name"], "tool_args": s.get("tool_args", {})} for s in trajectory if s.get("tool_name")
    ]
    if len(calls) != len(expected_calls):
        return False
    for call, expected in zip(calls, expected_calls, strict=True):
        if call["tool_name"] != expected["tool_name"]:
            return False
        missing_args = set(expected.get("tool_args", {})) - set(call["tool_args"])
        if missing_args:
            return False
    return True


# --- E13: goal completion + pass^k ------------------------------------------------


def final_state_match(actual: dict, expected: dict) -> bool:
    """tau-bench style: compare final environment/database state to the goal state."""
    return actual == expected


def pass_hat_k(tasks: list[list[bool]]) -> float:
    """pass^k = fraction of tasks that succeed on ALL k independent trials.
    `tasks` is a list of per-task trial-outcome lists (each of length k)."""
    if not tasks:
        return 0.0
    successes = sum(1 for trials in tasks if all(trials))
    return successes / len(tasks)


def run(fixture: dict) -> None:
    trajectory, expected = fixture["trajectory"], fixture["expected_trajectory"]

    section("E11: tool-call AST-style match (BFCL-style)")
    kv("ast_match(trajectory, expected_calls)", ast_match(trajectory, fixture["expected_calls"]))

    section("E12: trajectory match variants")
    kv("strict_match", strict_match(trajectory, expected))
    kv("in_order_match", in_order_match(trajectory, expected))
    kv("any_order_match", any_order_match(trajectory, expected))
    kv("precision/recall", tool_precision_recall(trajectory, expected))

    section("E13: goal completion & reliability")
    kv("final_state_match", final_state_match(fixture["actual_final_state"], fixture["expected_final_state"]))
    tasks = fixture["pass_hat_k_tasks"]
    kv(f"pass^{len(tasks[0])} across {len(tasks)} tasks", pass_hat_k(tasks))


if __name__ == "__main__":
    run(load_fixture())
