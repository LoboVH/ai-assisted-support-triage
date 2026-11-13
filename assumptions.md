# Pipeline assumptions and offline behavior

- The default `LLM_MODE=mock` runs all four LLM-shaped stages using local deterministic routines. It still writes one audit record for triage, response drafting, action planning, and grounding validation per ticket. No provider credentials or network calls are used in mock mode.
- The mock Stage 4 validator explicitly checks the deterministic claim records and reports whether those records already contain unsupported claims. It is a reproducible stage stub, not an independent model judgment.
- In `mock` and `off` modes, retrieval uses local token-count cosine similarity and claim checks use conservative local token/negation heuristics. These are fallback approximations, not transformer embeddings or a full natural-language-inference model.
- Weak retrieval is identified by empty evidence, a low top score, or a possible omitted relevant chunk. The pipeline sets `needs_human_review`, avoids policy-specific customer claims, and emits a generic human-review action plan.
- Live provider mode is selected with `LLM_MODE=live`. `LLM_PROVIDER` selects the preferred provider. Configure the corresponding provider credentials, such as `GROQ_API_KEY` or `GEMINI_API_KEY`. Live calls may use provider SDKs and credentials. If model initialization or a call fails, the pipeline records the relevant stage and provider-attempt information where available, then uses conservative fallbacks.
- `SUPPORT_TRIAGE_ROOT` can point to a directory containing replacement `tickets.json` and `knowledge_base/` fixtures. All generated artifacts and logs for that run are written under that root.
- Generated corpus and output files are runtime products. Their presence in a checkout is not used as evidence that a run regenerated them.
