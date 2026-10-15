"""Append-only audit trail for the document retention/archival lifecycle.

Records structured events (upload, ingestion, archival, retrieval,
cleanup_execution) through the application logger and as JSON lines under
``settings.logs_dir`` for later querying. Purely additive/observational -
never raises, so a logging failure can never block the ingestion, archival,
or retrieval flow it is observing.

Audit entries intentionally never include internal filesystem paths,
credentials, or infrastructure details (only ``document_id`` and small,
non-sensitive detail fields).

No UI code lives here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from config import settings
from utils.helpers import get_logger

logger = get_logger(__name__)

VALID_EVENTS = ("upload", "ingestion", "archival", "retrieval", "cleanup_execution")
_WRITE_LOCK = Lock()


def _audit_log_path() -> Path:
    return settings.logs_dir / "document_audit_log.jsonl"


def record_audit_event(event: str, document_id: str, details: dict | None = None) -> None:
    """Record one audit event for ``document_id``.

    ``event`` must be one of :data:`VALID_EVENTS`. Never raises - any failure
    to persist the entry is logged and swallowed.
    """
    if event not in VALID_EVENTS:
        raise ValueError(f"Unknown audit event: {event!r}")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "document_id": document_id,
        "details": details or {},
    }
    logger.info("AUDIT %s", entry)
    try:
        with _WRITE_LOCK:
            settings.logs_dir.mkdir(parents=True, exist_ok=True)
            with _audit_log_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
    except OSError:
        logger.exception("Failed to persist audit log entry for event %s", event)
