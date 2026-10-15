"""Persist in-progress Data Ingestion file uploads as "drafts" so the uploaded
file and its metadata survive sidebar page switches and full browser refreshes,
without requiring the user to re-upload unless the draft has failed, been
removed, expired, or already been ingested.

A draft is a plain JSON record (same local-storage convention as
``storage/json_storage.py``) plus the raw uploaded bytes cached on disk under
``settings.storage_dir / "drafts"``. The Streamlit UI identifies the active
draft for the current browser tab via a ``draft_id`` kept in
``st.query_params`` - unlike ``st.session_state``, query params survive a full
page refresh (same URL), which is what makes cross-refresh restore possible
without a login/user-account system.

No UI code lives here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import settings
from utils.helpers import get_logger, read_json, write_json

logger = get_logger(__name__)

DRAFTS_FILE = "ingestion_drafts.json"

VALID_STATUSES = ("uploaded", "ingested", "failed")


@dataclass
class Draft:
    """One in-progress (or recently finished) Data Ingestion upload."""

    draft_id: str
    file_name: str
    file_size: int
    content_type: str | None
    status: str  # "uploaded" | "ingested" | "failed"
    uploaded_at: str
    expires_at: str
    stored_path: str
    document_id: str | None = None
    classification: str | None = None

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > datetime.fromisoformat(self.expires_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "status": self.status,
            "uploaded_at": self.uploaded_at,
            "expires_at": self.expires_at,
            "stored_path": self.stored_path,
            "document_id": self.document_id,
            "classification": self.classification,
        }


def _drafts_dir() -> Path:
    directory = settings.storage_dir / "drafts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _drafts_record_path() -> Path:
    return settings.storage_dir / DRAFTS_FILE


def _load_all() -> dict[str, dict[str, Any]]:
    return read_json(_drafts_record_path(), default={}) or {}


def _save_all(drafts: dict[str, dict[str, Any]]) -> None:
    write_json(_drafts_record_path(), drafts)


def create_draft(file_name: str, file_size: int, content_type: str | None, raw_bytes: bytes) -> Draft:
    """Persist a newly-uploaded file's bytes + metadata as a new draft.

    Returns the created :class:`Draft` (its ``draft_id`` should be stored in
    ``st.query_params`` by the caller so it can be restored later).
    """
    draft_id = uuid.uuid4().hex
    suffix = Path(file_name).suffix
    stored_path = _drafts_dir() / f"{draft_id}{suffix}"
    stored_path.write_bytes(raw_bytes)

    now = datetime.now(timezone.utc)
    draft = Draft(
        draft_id=draft_id,
        file_name=file_name,
        file_size=file_size,
        content_type=content_type,
        status="uploaded",
        uploaded_at=now.isoformat(),
        expires_at=(now + timedelta(hours=settings.ingestion_draft_ttl_hours)).isoformat(),
        stored_path=str(stored_path),
    )
    drafts = _load_all()
    drafts[draft_id] = draft.as_dict()
    _save_all(drafts)
    logger.info("Created ingestion draft %s for %s (%d bytes)", draft_id, file_name, file_size)
    return draft


def get_draft(draft_id: str) -> Draft | None:
    """Return the draft for ``draft_id``, or ``None`` if it no longer exists."""
    if not draft_id:
        return None
    record = _load_all().get(draft_id)
    if record is None:
        return None
    return Draft(**record)


def read_draft_bytes(draft: Draft) -> bytes | None:
    """Read the cached raw bytes for ``draft``, or ``None`` if missing on disk
    (e.g. the drafts directory was cleared out-of-band)."""
    path = Path(draft.stored_path)
    if not path.exists():
        return None
    return path.read_bytes()


def update_draft_status(draft_id: str, status: str) -> None:
    """Update a draft's status in place (e.g. to ``"ingested"`` or ``"failed"``)."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid draft status: {status!r}")
    drafts = _load_all()
    if draft_id in drafts:
        drafts[draft_id]["status"] = status
        _save_all(drafts)


def mark_ingested(draft_id: str, document_id: str, classification: str | None) -> None:
    """Record that a draft's ingestion pipeline completed successfully.

    Unlike :func:`delete_draft`, this keeps the record (status ``"ingested"``)
    so returning to the Data Ingestion page later still shows the document was
    already uploaded/ingested instead of reverting to an empty upload state.
    """
    drafts = _load_all()
    if draft_id in drafts:
        drafts[draft_id]["status"] = "ingested"
        drafts[draft_id]["document_id"] = document_id
        drafts[draft_id]["classification"] = classification
        _save_all(drafts)


def delete_draft(draft_id: str) -> None:
    """Remove a draft's record and its cached bytes.

    Used when the upload failed, the user explicitly removed the file, or the
    ingestion pipeline completed successfully (nothing left to resume).
    """
    drafts = _load_all()
    record = drafts.pop(draft_id, None)
    _save_all(drafts)
    if record is not None:
        Path(record["stored_path"]).unlink(missing_ok=True)
    logger.info("Deleted ingestion draft %s", draft_id)
