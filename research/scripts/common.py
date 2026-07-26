"""Shared helpers for the eval-review demo scripts.

Every layer_*.py script can run standalone (`python3 layer_a_deterministic.py`)
or be imported by run_all.py. Both paths load the same fixture so results are
comparable across layers/scripts. See ../README.md for the full picture.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Sequence

FIXTURE_PATH = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "shared_fixture.json"

_TOKEN_RE = re.compile(r"\w+")


def load_fixture(path: pathlib.Path = FIXTURE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> list[tuple]:
    if len(tokens) < n:
        return []
    return list(zip(*[tokens[i:] for i in range(n)], strict=False))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def kv(label: str, value) -> None:
    if isinstance(value, float):
        print(f"{label}: {value:.4f}")
    else:
        print(f"{label}: {value}")
