"""Text embeddings for semantic news search.

The default ``HashingEmbedder`` is dependency-free (uses scikit-learn, already a
dep) and deterministic — a lexical baseline good enough to wire up the full
vector-search pipeline and tests without pulling in heavyweight models. For
genuinely *semantic* embeddings, install ``sentence-transformers`` and use
``SentenceTransformerEmbedder`` (its all-MiniLM-L6-v2 model is also 384-dim, so
it drops into the same DB column). The Embedder interface and the
``NewsArticle.embedding`` column stay identical either way.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

EMBED_DIM = 384  # matches sentence-transformers/all-MiniLM-L6-v2


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic L2-normalized hashed bag-of-words vectors (lexical baseline).

    Not truly semantic (no learned meaning), but a real, fixed-dim vector space
    that powers cosine search and is dependency-free. Swap in
    ``SentenceTransformerEmbedder`` for semantic quality.
    """

    name = "hashing"

    def __init__(self, dim: int = EMBED_DIM) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")

    def encode(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vec.transform(list(texts))
        return matrix.toarray().astype("float32").tolist()


class SentenceTransformerEmbedder:  # pragma: no cover - optional heavy dep
    """Semantic embeddings via sentence-transformers (lazy import; optional)."""

    name = "sentence-transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()


def get_default_embedder() -> Embedder:
    """The dependency-free default. Override at call sites to use a real model."""
    return HashingEmbedder()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; 0 if either vector is zero-length."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
