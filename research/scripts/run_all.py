"""Run every layer's demo against the shared fixture in one pass.

    python3 research/scripts/run_all.py

Each layer_*.py module also runs standalone (`python3 layer_d_rag.py`) if you
only want to poke at one layer. See ../README.md for the full method index.
"""

from __future__ import annotations

import layer_a_deterministic
import layer_b_semantic
import layer_c_llm_judge
import layer_d_rag
import layer_e_agent
import layer_f_safety
import layer_g_human_meta
import layer_h_stats_monitoring
from common import load_fixture

LAYERS = [
    ("Layer A -- Deterministic / reference-based checks", layer_a_deterministic),
    ("Layer B -- Semantic & model-based metrics", layer_b_semantic),
    ("Layer C -- LLM-as-judge family", layer_c_llm_judge),
    ("Layer D -- RAG-specific", layer_d_rag),
    ("Layer E -- Agent-specific", layer_e_agent),
    ("Layer F -- Safety / security / adversarial", layer_f_safety),
    ("Layer G -- Human & meta-evaluation", layer_g_human_meta),
    ("Layer H -- Online monitoring & statistical rigor", layer_h_stats_monitoring),
]


def main() -> None:
    fixture = load_fixture()
    for title, module in LAYERS:
        print(f"\n{'#' * 78}\n# {title}\n{'#' * 78}")
        module.run(fixture)


if __name__ == "__main__":
    main()
