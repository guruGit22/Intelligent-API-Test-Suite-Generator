"""Root pytest configuration.

This file lives in ``tests/`` (one level below the project root). It explicitly
adds the project root and ``src/`` to ``sys.path`` so that ``from config import
settings`` and ``from ingestion...``/``from parser...`` style imports used
throughout the codebase resolve correctly during test collection regardless of
the current working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES_DIR


@pytest.fixture
def petstore_spec_path() -> Path:
    return SAMPLES_DIR / "petstore.yaml"


@pytest.fixture
def mongo_repository(monkeypatch):
    """A MongoDocumentRepository backed by ``mongomock`` instead of a real server."""
    import mongomock

    from ingestion import mongo_document_tool

    monkeypatch.setattr(mongo_document_tool, "MongoClient", mongomock.MongoClient)
    return mongo_document_tool.MongoDocumentRepository(uri="mongodb://localhost:27017")


class _FakeSentenceTransformer:
    """Deterministic, offline stand-in for ``sentence_transformers.SentenceTransformer``.

    Avoids downloading real embedding weights during tests while still exercising
    the real FAISS build/save/load/query code paths in ``DocumentVectorStore``.
    """

    _DIM = 16

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        import numpy as np

        vectors = np.zeros((len(texts), self._DIM), dtype="float32")
        for row, text in enumerate(texts):
            for i, ch in enumerate(text.encode("utf-8")):
                vectors[row, i % self._DIM] += ch
        return vectors


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    """Patch ``SentenceTransformer`` in the document vector store module."""
    from ingestion import document_vector_store

    monkeypatch.setattr(document_vector_store, "SentenceTransformer", _FakeSentenceTransformer)
    return _FakeSentenceTransformer
