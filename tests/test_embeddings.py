"""Tests for slow_wave.embeddings: HashEmbedder determinism + embedding_sha256."""

from __future__ import annotations

import numpy as np

from slow_wave.embeddings import HashEmbedder, embed_texts, embedding_sha256


def test_encode_shape_and_dtype() -> None:
    """encode returns an (n, dim) float32 array."""
    emb = HashEmbedder(dim=384)
    texts = ["hello world", "memory consolidation", "the quick brown fox"]
    arr = emb.encode(texts)

    assert arr.shape == (3, 384)
    assert arr.dtype == np.float32


def test_rows_are_l2_normalized() -> None:
    """Non-empty rows have unit L2 norm."""
    emb = HashEmbedder(dim=64)
    arr = emb.encode(["alpha beta gamma", "single"])
    norms = np.linalg.norm(arr, axis=1)

    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_empty_text_is_zero_vector() -> None:
    """Texts with no alphanumeric tokens map to the zero vector."""
    emb = HashEmbedder(dim=32)
    arr = emb.encode(["", "   ", "!!! ??? ..."])
    norms = np.linalg.norm(arr, axis=1)

    np.testing.assert_allclose(norms, 0.0, atol=1e-7)


def test_deterministic_within_instance() -> None:
    """Two encode calls on one instance are byte-identical."""
    emb = HashEmbedder(dim=128)
    texts = ["reproducible vectors", "slow wave bench"]

    a = emb.encode(texts)
    b = emb.encode(texts)
    assert a.tobytes() == b.tobytes()


def test_deterministic_across_fresh_instances() -> None:
    """Two fresh instances produce byte-identical output for the same input."""
    texts = ["reproducible vectors", "slow wave bench"]

    a = HashEmbedder(dim=128).encode(texts)
    b = HashEmbedder(dim=128).encode(texts)
    assert a.tobytes() == b.tobytes()


def test_different_texts_give_different_vectors() -> None:
    """Distinct texts produce distinct embeddings."""
    emb = HashEmbedder(dim=128)
    arr = emb.encode(["cat", "dog"])

    assert arr[0].tobytes() != arr[1].tobytes()


def test_embedding_sha256_stable_for_same_array() -> None:
    """The same embedding content hashes to the same digest."""
    emb = HashEmbedder(dim=64)
    a = emb.encode(["one", "two"])
    b = emb.encode(["one", "two"])

    digest = embedding_sha256(a)
    assert isinstance(digest, str)
    assert len(digest) == 64  # sha256 hex
    assert digest == embedding_sha256(b)


def test_embedding_sha256_changes_for_different_arrays() -> None:
    """Different embedding content hashes to a different digest."""
    emb = HashEmbedder(dim=64)
    a = emb.encode(["one", "two"])
    c = emb.encode(["one", "three"])

    assert embedding_sha256(a) != embedding_sha256(c)


def test_embed_texts_wrapper_is_2d_float32() -> None:
    """embed_texts always returns a 2D float32 array."""
    emb = HashEmbedder(dim=16)
    arr = embed_texts(emb, ["x y z"])

    assert arr.ndim == 2
    assert arr.shape == (1, 16)
    assert arr.dtype == np.float32
