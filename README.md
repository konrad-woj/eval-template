# eval-template
Evaluation pipeline template built using best practices

## Eval methodology research

[`research/`](research/README.md) holds a reference catalog of 18 AI/LLM eval
methodologies (BLEU/ROUGE, G-Eval, LLM-as-judge, RAG retrieval/faithfulness,
agent trajectory/tool-call/goal-completion, safety, judge-alignment stats,
production monitoring) plus runnable, dependency-free demo scripts for each.
It's a research library, not part of the eval pipeline itself — start there
when picking an eval approach for a new feature.
