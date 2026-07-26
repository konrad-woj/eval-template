"""Layer F -- Safety / security / adversarial.

F14. Safety classifiers -- toy regex-based PII and keyword-based toxicity
     detectors. Real production use should use Presidio (PII), Detoxify
     (toxicity), or Llama Guard -- see ../CATALOG.md#layer-f. The maintainers
     of Presidio themselves caveat "there is no guarantee that Presidio will
     find all sensitive information"; these demos are strictly weaker than
     that and exist only to show the interface shape.
F15. Adversarial & prompt-injection heuristics -- pattern match against a
     small set of known injection phrasings, mapped to OWASP LLM01. Real
     red-teaming should use promptfoo red-team, PyRIT, or garak, and draw on
     benchmark suites like AgentDojo/AgentHarm/InjecAgent.

Stdlib only.
"""

from __future__ import annotations

import re

from common import load_fixture, section, tokenize

_PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phone": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

_TOXIC_KEYWORDS = {"idiot", "stupid", "hate", "dumb", "shut up"}

_INJECTION_PATTERNS = [
    r"ignore (all |any )?previous instructions",
    r"disregard (the )?(system|prior) prompt",
    r"reveal (your|the) (system )?prompt",
    r"you are now\b",
    r"forget (everything|all) (you were told|above)",
]


def detect_pii(text: str) -> list[dict]:
    spans = []
    for label, pattern in _PII_PATTERNS.items():
        for m in re.finditer(pattern, text):
            spans.append({"type": label, "text": m.group(), "start": m.start(), "end": m.end()})
    return spans


def toxicity_score(text: str) -> float:
    """Crude keyword-hit-rate proxy in [0, 1]; NOT a substitute for Detoxify."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _TOXIC_KEYWORDS)
    return min(1.0, hits / len(tokens) * 10)


def detect_prompt_injection(text: str) -> dict:
    """OWASP LLM01 (Prompt Injection) pattern-match heuristic."""
    matched = [p for p in _INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]
    return {"flagged": bool(matched), "matched_patterns": matched}


def run(fixture: dict) -> None:
    section("F14: safety classifiers (toy PII + toxicity)")
    for text in fixture["safety_test_inputs"]:
        pii = detect_pii(text)
        tox = toxicity_score(text)
        print(f"  {text!r}")
        print(f"    pii_spans={pii}")
        print(f"    toxicity={tox:.3f}")

    section("F15: prompt-injection heuristic (OWASP LLM01)")
    for text in fixture["safety_test_inputs"]:
        result = detect_prompt_injection(text)
        flag = "FLAGGED" if result["flagged"] else "clean"
        print(f"  [{flag}] {text!r} -> {result['matched_patterns']}")


if __name__ == "__main__":
    run(load_fixture())
