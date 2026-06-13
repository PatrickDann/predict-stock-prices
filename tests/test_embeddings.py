"""Embedder behavior (lexical baseline) + cosine helper."""

from market_intel.embeddings import EMBED_DIM, HashingEmbedder, cosine_similarity


def test_dim_and_shape():
    emb = HashingEmbedder()
    assert emb.dim == EMBED_DIM
    vecs = emb.encode(["oil supply shock", "central bank"])
    assert len(vecs) == 2
    assert all(len(v) == EMBED_DIM for v in vecs)


def test_deterministic():
    emb = HashingEmbedder()
    assert emb.encode(["same text"]) == emb.encode(["same text"])


def test_similar_text_scores_higher():
    emb = HashingEmbedder()
    base, similar, different = emb.encode(
        [
            "oil prices surge on supply fears",
            "oil supply fears push prices up",
            "local bakery wins award",
        ]
    )
    assert cosine_similarity(base, similar) > cosine_similarity(base, different)


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
