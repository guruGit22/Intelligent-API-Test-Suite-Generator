"""Tests for ``storage/retention_store.py`` - the document retention/archival
lifecycle (Active -> Archived -> Retrieved / Deleted) for originally uploaded
Data Ingestion documents. Independent of MongoDB/vector-store behavior, which
this module never touches.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from storage import retention_store as store


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    from config import settings as global_settings

    isolated = dataclasses.replace(
        global_settings,
        storage_dir=tmp_path,
        document_archive_active_dir=tmp_path / "active",
        document_archive_compressed_dir=tmp_path / "compressed",
        document_retention_minutes=15,
    )
    monkeypatch.setattr(store, "settings", isolated)


def test_register_document_creates_active_record():
    record = store.register_document("doc1", "spec.yaml", b"hello world")

    assert record.status == "Active"
    assert record.document_id == "doc1"
    assert not record.is_expired


def test_get_record_round_trips_a_registered_document():
    store.register_document("doc1", "spec.yaml", b"hello world")

    fetched = store.get_record("doc1")

    assert fetched is not None
    assert fetched.file_name == "spec.yaml"
    assert fetched.status == "Active"


def test_get_record_returns_none_for_unknown_document():
    assert store.get_record("does-not-exist") is None


def test_archive_expired_does_nothing_before_retention_elapses():
    store.register_document("doc1", "spec.yaml", b"hello world")

    archived = store.archive_expired()

    assert archived == []
    assert store.get_record("doc1").status == "Active"


def test_archive_expired_compresses_and_updates_status(monkeypatch):
    record = store.register_document("doc1", "spec.yaml", b"hello world")
    # Force the record into the past so it is due for archival.
    records = store._load_all()
    records["doc1"]["retention_until"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    store._save_all(records)

    archived = store.archive_expired()

    assert len(archived) == 1
    updated = store.get_record("doc1")
    assert updated.status == "Archived"
    assert updated.archive_path is not None
    assert updated.active_path is None
    from pathlib import Path

    assert Path(updated.archive_path).exists()
    assert Path(updated.archive_path).suffix == ".zip"


def test_retrieve_archived_document_returns_original_bytes_and_marks_retrieved():
    store.register_document("doc1", "spec.yaml", b"hello world")
    records = store._load_all()
    records["doc1"]["retention_until"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    store._save_all(records)
    store.archive_expired()

    data = store.retrieve_archived_document("doc1")

    assert data == b"hello world"
    assert store.get_record("doc1").status == "Retrieved"


def test_retrieve_archived_document_returns_none_for_unknown_document():
    assert store.retrieve_archived_document("does-not-exist") is None


def test_mark_deleted_removes_archive_and_updates_status():
    store.register_document("doc1", "spec.yaml", b"hello world")
    records = store._load_all()
    records["doc1"]["retention_until"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    store._save_all(records)
    store.archive_expired()

    store.mark_deleted("doc1")

    updated = store.get_record("doc1")
    assert updated.status == "Deleted"
    assert updated.archive_path is None


def test_list_records_returns_all_registered_documents():
    store.register_document("doc1", "spec.yaml", b"hello world")
    store.register_document("doc2", "other.json", b"{}")

    records = {r.document_id for r in store.list_records()}

    assert records == {"doc1", "doc2"}
