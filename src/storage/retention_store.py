"""Enterprise document retention & archival lifecycle for Data Ingestion uploads.

Tracks each ingested document's *original uploaded file* through a lifecycle of
``Active -> Archived -> Retrieved`` (or ``Deleted``), independent of the
existing MongoDB document/chunk records and FAISS vector index, which remain
the sole source of truth for metadata, chunks, and embeddings and are never
touched by this module. Ingestion, retrieval, and test-case generation keep
working exactly as before regardless of a document's retention status.

Retention duration, archive location, and enablement are all configurable via
``config.settings`` (environment variables / ``.env``), following the same
pattern as ``settings.ingestion_draft_ttl_hours``.

No UI code lives here.
"""
from __future__ import annotations

import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import settings
from utils.helpers import get_logger, read_json, write_json

logger = get_logger(__name__)

RECORDS_FILE = "document_retention.json"
VALID_STATUSES = ("Active", "Archived", "Retrieved", "Deleted")


@dataclass
class RetentionRecord:
    """One document's retention/archival lifecycle state.

    ``active_path``/``archive_path`` are internal filesystem locations and
    must never be surfaced directly to the UI (only status/timestamps are).
    """

    document_id: str
    file_name: str
    status: str
    ingested_at: str
    retention_until: str
    active_path: str | None = None
    archive_path: str | None = None
    archived_at: str | None = None
    retrieved_at: str | None = None
    deleted_at: str | None = None

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > datetime.fromisoformat(self.retention_until)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _records_path() -> Path:
    return settings.storage_dir / RECORDS_FILE


def _active_dir() -> Path:
    settings.document_archive_active_dir.mkdir(parents=True, exist_ok=True)
    return settings.document_archive_active_dir


def _archive_dir() -> Path:
    settings.document_archive_compressed_dir.mkdir(parents=True, exist_ok=True)
    return settings.document_archive_compressed_dir


def _load_all() -> dict[str, dict[str, Any]]:
    return read_json(_records_path(), default={}) or {}


def _save_all(records: dict[str, dict[str, Any]]) -> None:
    write_json(_records_path(), records)


def register_document(document_id: str, file_name: str, raw_bytes: bytes) -> RetentionRecord:
    """Start the retention lifecycle for a just-ingested document's original
    file. Re-registering the same ``document_id`` replaces its prior record
    with a fresh ``Active`` one (used if the same document is re-ingested).
    """
    suffix = Path(file_name).suffix
    stored_path = _active_dir() / f"{document_id}{suffix}"
    stored_path.write_bytes(raw_bytes)

    now = datetime.now(timezone.utc)
    record = RetentionRecord(
        document_id=document_id,
        file_name=file_name,
        status="Active",
        ingested_at=now.isoformat(),
        retention_until=(now + timedelta(minutes=settings.document_retention_minutes)).isoformat(),
        active_path=str(stored_path),
    )
    records = _load_all()
    records[document_id] = record.as_dict()
    _save_all(records)
    logger.info("Retention registered for document %s (retain until %s)", document_id, record.retention_until)
    return record


def get_record(document_id: str) -> RetentionRecord | None:
    """Return the retention record for ``document_id``, or ``None`` if it was
    never registered (e.g. retention was disabled at ingestion time)."""
    raw = _load_all().get(document_id)
    return RetentionRecord(**raw) if raw is not None else None


def list_records() -> list[RetentionRecord]:
    """All known retention records, for status-tracking/display purposes."""
    return [RetentionRecord(**raw) for raw in _load_all().values()]


def archive_expired() -> list[RetentionRecord]:
    """Compress and move every ``Active`` record whose retention period has
    elapsed into the archive directory (ZIP, then delete the original).

    Returns the records archived in this pass (empty if none were due yet).
    Intended to be called periodically by the background scheduler, but is a
    plain synchronous function so it can also be called directly/tested.
    """
    records = _load_all()
    archived: list[RetentionRecord] = []
    now = datetime.now(timezone.utc)

    for document_id, raw in records.items():
        record = RetentionRecord(**raw)
        if record.status != "Active" or not record.is_expired:
            continue

        active_path = Path(record.active_path) if record.active_path else None
        if active_path is None or not active_path.exists():
            logger.warning("Retention: active file missing for document %s; marking Deleted", document_id)
            record.status = "Deleted"
            record.deleted_at = now.isoformat()
            records[document_id] = record.as_dict()
            continue

        archive_path = _archive_dir() / f"{document_id}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(active_path, arcname=active_path.name)
        active_path.unlink(missing_ok=True)

        record.status = "Archived"
        record.archive_path = str(archive_path)
        record.archived_at = now.isoformat()
        record.active_path = None
        records[document_id] = record.as_dict()
        archived.append(record)
        logger.info("Archived document %s", document_id)

    if archived:
        _save_all(records)
    return archived


def retrieve_archived_document(document_id: str) -> bytes | None:
    """Authorized retrieval API: return a document's original bytes (from the
    active copy if still within retention, or decompressed from the archive
    otherwise). Returns ``None`` if the document is unknown or was deleted.

    Marks the record ``Retrieved`` on a successful archive read (the archive
    itself is left intact so it remains retrievable again later).
    """
    records = _load_all()
    raw = records.get(document_id)
    if raw is None:
        return None
    record = RetentionRecord(**raw)

    if record.status == "Active" and record.active_path:
        active_path = Path(record.active_path)
        return active_path.read_bytes() if active_path.exists() else None

    if record.status not in ("Archived", "Retrieved") or not record.archive_path:
        return None

    archive_path = Path(record.archive_path)
    if not archive_path.exists():
        return None
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if not names:
            return None
        data = zf.read(names[0])

    record.status = "Retrieved"
    record.retrieved_at = datetime.now(timezone.utc).isoformat()
    records[document_id] = record.as_dict()
    _save_all(records)
    logger.info("Retrieved archived document %s", document_id)
    return data


def mark_deleted(document_id: str) -> None:
    """Permanently remove an archived document's compressed file per
    retention policy, keeping the metadata record (status ``Deleted``) for
    traceability."""
    records = _load_all()
    raw = records.get(document_id)
    if raw is None:
        return
    record = RetentionRecord(**raw)
    if record.archive_path:
        Path(record.archive_path).unlink(missing_ok=True)
    record.archive_path = None
    record.active_path = None
    record.status = "Deleted"
    record.deleted_at = datetime.now(timezone.utc).isoformat()
    records[document_id] = record.as_dict()
    _save_all(records)
    logger.info("Deleted archived document %s", document_id)
