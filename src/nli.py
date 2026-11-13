import os
import re


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


class NLIModel:

    def __init__(self):
        self.model = None

        nli_mode = os.getenv("NLI_MODE", "").strip().lower()
        llm_mode = os.getenv("LLM_MODE", "mock").strip().lower()

        if nli_mode and nli_mode not in {"mock", "off", "live"}:
            raise ValueError(
                f"Unsupported NLI_MODE: {nli_mode}. "
                "Expected 'mock', 'off', or 'live'."
            )

        forced_offline = (
            os.getenv("SUPPORT_TRIAGE_OFFLINE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        if forced_offline:
            self.offline = True
        elif nli_mode:
            self.offline = nli_mode in {"mock", "off"}
        else:
            self.offline = llm_mode in {"mock", "off"}

        if not self.offline:
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(MODEL_NAME)
            except Exception as exc:
                self.offline = True
                print(
                    "NLI model unavailable; offline claims will remain "
                    f"unverified: {type(exc).__name__}: {exc}"
                )


    # def predict(self, pairs):
    #     if self.model is not None:
    #         return self.model.predict(pairs)

    #     import numpy as np

    #     predictions = []
    #     for premise, hypothesis in pairs:
    #         premise_tokens = set(re.findall(r"[a-z0-9]+", str(premise).casefold()))
    #         hypothesis_tokens = set(re.findall(r"[a-z0-9]+", str(hypothesis).casefold()))
    #         overlap = len(premise_tokens & hypothesis_tokens)
    #         coverage = overlap / max(len(hypothesis_tokens), 1)
    #         premise_negated = bool(re.search(r"\b(?:not|never|no|cannot|can't|isn't|wasn't|won't)\b", str(premise).casefold()))
    #         hypothesis_negated = bool(re.search(r"\b(?:not|never|no|cannot|can't|isn't|wasn't|won't)\b", str(hypothesis).casefold()))

    #         if coverage >= 0.92 and premise_negated == hypothesis_negated:
    #             predictions.append([0.0, 5.0, 0.0])
    #         elif coverage >= 0.92 and premise_negated != hypothesis_negated:
    #             predictions.append([5.0, 0.0, 0.0])
    #         else:
    #             predictions.append([0.0, 0.0, 5.0])

    #     return np.asarray(predictions, dtype=float)

    def predict(self, pairs):
        if self.model is not None:
            return self.model.predict(pairs)

        import numpy as np

        # Lexical overlap is not sufficient to establish entailment
        # or contradiction. Keep offline NLI claims unverified.
        return np.zeros((len(pairs), 3), dtype=float)
