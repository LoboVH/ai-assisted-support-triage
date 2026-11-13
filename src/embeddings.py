import math
import os
import re
from collections import Counter


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self):
        self.model = None
        self.offline = (
            os.getenv("SUPPORT_TRIAGE_OFFLINE", "").lower() in {"1", "true", "yes"}
            or os.getenv("LLM_MODE", "mock").lower() in {"mock", "off"}
        )

        if not self.offline:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(MODEL_NAME)
            except Exception as exc:
                self.offline = True
                print(
                    "Embedding model unavailable; using deterministic local "
                    f"token similarity: {type(exc).__name__}: {exc}"
                )

    def encode(self, text: str):
        if self.model is not None:
            return self.model.encode(text)

        if isinstance(text, str):
            return [self._vector(text)]
        return [self._vector(item) for item in text]

    def similarity(self, embeddings1, embeddings2):
        if self.model is not None:
            return self.model.similarity(embeddings1, embeddings2)

        import numpy as np

        left = embeddings1 if isinstance(embeddings1, list) else [embeddings1]
        right = embeddings2 if isinstance(embeddings2, list) else [embeddings2]
        if left and isinstance(left[0], dict):
            left_vectors = left
        else:
            left_vectors = [left]
        if right and isinstance(right[0], dict):
            right_vectors = right
        else:
            right_vectors = [right]

        return np.asarray([
            [self._cosine(a, b) for b in right_vectors]
            for a in left_vectors
        ], dtype=float)

    @staticmethod
    def _vector(text):
        tokens = re.findall(r"[a-z0-9]+", str(text).casefold())
        return dict(Counter(tokens))

    @staticmethod
    def _cosine(left, right):
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(token, 0) for token, value in left.items())
        norm_left = math.sqrt(sum(value * value for value in left.values()))
        norm_right = math.sqrt(sum(value * value for value in right.values()))
        return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0
