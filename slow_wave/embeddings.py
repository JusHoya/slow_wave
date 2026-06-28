"""Embedding backends for the Slow Wave bench (Phase 0).

Two backends share a small duck-typed interface so the rest of the bench can
treat them interchangeably and record their provenance in the run manifest:

* attributes ``backend: str``, ``model: str``, ``version: str``, ``dim: int``
* method ``encode(texts: list[str]) -> np.ndarray`` returning an
  ``(n, dim)`` ``float32`` array of L2-normalized row vectors.

:class:`HashEmbedder` is dependency-free and **deterministic bit-for-bit across
runs, processes, and platforms** (it seeds NumPy's PCG64 from a BLAKE2b digest
of each token rather than Python's salted ``hash()``). It is the default for the
Phase 0 smoke run so CI stays green with no heavy ML dependencies.

:class:`SentenceTransformerEmbedder` lazily imports ``sentence-transformers`` and
is selected only when a config explicitly requests it; if the import or model
load fails, :func:`get_embedder` transparently falls back to
:class:`HashEmbedder` and records why on ``.fallback_reason``.
"""

from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# Tokenizer shared by the hash backend: lowercase alphanumeric runs.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic, dependency-free bag-of-words hashing embedder.

    Each text is lowercased and tokenized into alphanumeric runs. Every token
    seeds a private ``numpy.random.default_rng`` (PCG64) from a BLAKE2b digest of
    the token bytes; the RNG's ``standard_normal(dim)`` vector is accumulated
    (bag-of-words sum) and the row is finally L2-normalized. Because BLAKE2b and
    PCG64 are platform-independent, the output is reproducible bit-for-bit
    across runs, processes, and operating systems.

    Empty texts (no alphanumeric tokens) map to the zero vector.
    """

    def __init__(self, dim: int = 384) -> None:
        """Initialize the embedder.

        Args:
            dim: Output embedding dimensionality (number of columns).
        """
        self.backend: str = "hash"
        self.model: str = "hash-bow-v1"
        self.version: str = "1.0"
        self.dim: int = int(dim)

    def _embed_one(self, text: str) -> np.ndarray:
        """Embed a single text into a ``(dim,)`` float64 accumulator (unnormalized)."""
        acc = np.zeros(self.dim, dtype=np.float64)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            seed = int.from_bytes(digest, "big")
            acc += np.random.default_rng(seed).standard_normal(self.dim)
        return acc

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into an ``(n, dim)`` float32 array of L2-normalized rows.

        Args:
            texts: The texts to embed.

        Returns:
            A ``(len(texts), dim)`` ``float32`` array. Each non-empty row has unit
            L2 norm; rows for texts with no alphanumeric tokens are all zeros.
        """
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            acc = self._embed_one(text)
            norm = float(np.linalg.norm(acc))
            if norm > 0.0:
                acc = acc / norm
            out[i] = acc
        return out.astype(np.float32)


class SentenceTransformerEmbedder:
    """Local ``sentence-transformers`` embedder (lazy import; heavy deps).

    Loads a :class:`~sentence_transformers.SentenceTransformer` model and returns
    normalized embeddings. The model's own
    ``get_sentence_embedding_dimension()`` is authoritative for ``dim`` (the
    ``dim`` constructor argument is accepted for signature symmetry with
    :class:`HashEmbedder` but does not override the model).
    """

    def __init__(self, model_name: str, dim: int | None = None) -> None:
        """Load the named sentence-transformers model.

        Args:
            model_name: Hugging Face model id, e.g. ``"BAAI/bge-small-en-v1.5"``.
            dim: Ignored except as a hint; the model's reported embedding
                dimension is used.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
            Exception: Any error raised while loading the model.
        """
        import sentence_transformers  # lazy: heavy import, optional dependency
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.backend: str = "sentence-transformers"
        self.model: str = model_name
        self.version: str = sentence_transformers.__version__
        model_dim = self._model.get_sentence_embedding_dimension()
        self.dim: int = int(model_dim if model_dim is not None else (dim or 0))

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into an ``(n, dim)`` float32 array of normalized rows.

        Args:
            texts: The texts to embed.

        Returns:
            A ``(len(texts), dim)`` ``float32`` array of L2-normalized rows.
        """
        arr = self._model.encode(list(texts), normalize_embeddings=True)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


def get_embedder(cfg):
    """Construct the embedder requested by ``cfg.embedding.backend``.

    If the sentence-transformers backend is requested but cannot be constructed
    (missing ``sentence-transformers``/``torch``, or any model-load error), this
    falls back to a :class:`HashEmbedder` and records the reason on the returned
    embedder's ``.fallback_reason`` attribute (also logged at WARNING level), so
    the smoke run never hard-fails for want of heavy ML dependencies.

    Args:
        cfg: A :class:`slow_wave.config.Config`.

    Returns:
        An embedder exposing ``.backend``, ``.model``, ``.version``, ``.dim``,
        and ``.encode``.
    """
    if cfg.embedding.backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(cfg.embedding.model, cfg.embedding.dim)
        except Exception as exc:  # ImportError or any model-load failure
            embedder = HashEmbedder(cfg.embedding.dim)
            reason = (
                f"sentence-transformers backend '{cfg.embedding.model}' "
                f"unavailable ({type(exc).__name__}: {exc}); falling back to "
                f"hash backend."
            )
            embedder.fallback_reason = reason
            logger.warning(reason)
            return embedder
    return HashEmbedder(cfg.embedding.dim)


def embed_texts(embedder, texts: list[str]) -> np.ndarray:
    """Encode ``texts`` with ``embedder``, guaranteeing a 2D ``float32`` array.

    A thin wrapper over ``embedder.encode`` that normalizes the return shape and
    dtype so callers never have to special-case a 1D result.

    Args:
        embedder: Any object with an ``encode(texts) -> np.ndarray`` method.
        texts: The texts to embed.

    Returns:
        A 2D ``(n, dim)`` ``float32`` array.
    """
    arr = np.asarray(embedder.encode(texts))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32, copy=False)


def embedding_sha256(arr: np.ndarray) -> str:
    """Return a stable SHA-256 hex digest of an embedding array.

    The array is cast to ``float64`` and rounded to 6 decimal places before
    hashing. Rounding makes the digest robust to sub-ULP float noise across
    backends/platforms while remaining exactly reproducible for the
    deterministic :class:`HashEmbedder`.

    Args:
        arr: An embedding array.

    Returns:
        The 64-character SHA-256 hex digest of the rounded array's raw bytes.
    """
    rounded = np.round(arr.astype(np.float64), 6)
    return hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()
