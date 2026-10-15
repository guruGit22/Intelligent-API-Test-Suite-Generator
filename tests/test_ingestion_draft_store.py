"""Tests for ``storage/ingestion_draft_store.py`` - persistence of in-progress
Data Ingestion uploads so the uploaded file/metadata survive page navigation
and browser refresh without requiring re-upload (unless failed, removed,
expired, or already ingested).
"""
from __future__ import annotations

import dataclasses

import pytest

from storage import ingestion_draft_store as store


@pytest.fixture(autouse=True)
def _isolated_storage_dir(monkeypatch, tmp_path):
    from config import settings as global_settings

    isolated = dataclasses.replace(global_settings, storage_dir=tmp_path)
    monkeypatch.setattr(store, "settings", isolated)


def test_create_draft_persists_bytes_and_metadata():
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    assert draft.file_name == "spec.yaml"
    assert draft.file_size == 11
    assert draft.status == "uploaded"
    assert not draft.is_expired


def test_get_draft_round_trips_a_created_draft():
    created = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    fetched = store.get_draft(created.draft_id)

    assert fetched is not None
    assert fetched.draft_id == created.draft_id
    assert fetched.file_name == "spec.yaml"
    assert fetched.status == "uploaded"


def test_get_draft_returns_none_for_unknown_id():
    assert store.get_draft("does-not-exist") is None


def test_get_draft_returns_none_for_empty_id():
    assert store.get_draft("") is None


def test_read_draft_bytes_returns_original_content():
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    assert store.read_draft_bytes(draft) == b"hello world"


def test_read_draft_bytes_returns_none_if_file_missing_on_disk():
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")
    from pathlib import Path

    Path(draft.stored_path).unlink()

    assert store.read_draft_bytes(draft) is None


def test_update_draft_status_changes_status():
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    store.update_draft_status(draft.draft_id, "ingested")

    assert store.get_draft(draft.draft_id).status == "ingested"


def test_update_draft_status_rejects_invalid_status():
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    with pytest.raises(ValueError):
        store.update_draft_status(draft.draft_id, "not-a-real-status")


def test_mark_ingested_records_status_document_id_and_classification():
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    store.mark_ingested(draft.draft_id, "abc123", "api_specification")

    updated = store.get_draft(draft.draft_id)
    assert updated.status == "ingested"
    assert updated.document_id == "abc123"
    assert updated.classification == "api_specification"


def test_mark_ingested_draft_is_not_deleted_and_can_be_restored_later():
    """Simulates: ingest -> navigate away -> return - the draft (and its
    draft_id) must still resolve to the ingested document, not disappear."""
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")
    store.mark_ingested(draft.draft_id, "abc123", "api_specification")

    restored = store.get_draft(draft.draft_id)

    assert restored is not None
    assert restored.status == "ingested"
    assert restored.file_name == "spec.yaml"



def test_delete_draft_removes_record_and_file():
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    store.delete_draft(draft.draft_id)

    from pathlib import Path

    assert store.get_draft(draft.draft_id) is None
    assert not Path(draft.stored_path).exists()


def test_multiple_file_uploads_create_independent_drafts():
    first = store.create_draft("a.yaml", 5, "application/x-yaml", b"first")
    second = store.create_draft("b.json", 6, "application/json", b"second")

    assert first.draft_id != second.draft_id
    assert store.get_draft(first.draft_id).file_name == "a.yaml"
    assert store.get_draft(second.draft_id).file_name == "b.json"
    assert store.read_draft_bytes(store.get_draft(first.draft_id)) == b"first"
    assert store.read_draft_bytes(store.get_draft(second.draft_id)) == b"second"


def test_draft_expires_after_ttl(monkeypatch):
    import dataclasses as dc

    expired_settings = dc.replace(store.settings, ingestion_draft_ttl_hours=0)
    monkeypatch.setattr(store, "settings", expired_settings)

    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    assert draft.is_expired is True


def test_draft_not_expired_within_ttl():
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")
    assert draft.is_expired is False


def test_draft_resume_flow_end_to_end():
    """Simulates: upload -> navigate away -> return -> restore -> complete."""
    draft = store.create_draft("spec.yaml", 11, "application/x-yaml", b"hello world")

    # "Navigate away and back": look the draft up again by id only, as the UI
    # does via st.query_params, without holding a Python reference around.
    restored = store.get_draft(draft.draft_id)
    assert restored is not None
    assert restored.status == "uploaded"
    assert store.read_draft_bytes(restored) == b"hello world"

    # Draft knowledge base creation completes.
    store.update_draft_status(restored.draft_id, "ingested")
    completed = store.get_draft(restored.draft_id)
    assert completed.status == "ingested"
