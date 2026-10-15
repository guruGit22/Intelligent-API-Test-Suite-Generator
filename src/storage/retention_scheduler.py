"""Background scheduler for the document retention/archival cleanup job.

Runs ``retention_store.archive_expired()`` periodically on a single daemon
thread per process, so archival happens asynchronously and never blocks or
slows down the Data Ingestion upload/parse/chunk/embed flow. Starting the
scheduler is idempotent - safe to call on every Streamlit rerun - and it is a
no-op entirely when ``settings.document_retention_enabled`` is false.

No UI code lives here.
"""
from __future__ import annotations

import threading
import time

from config import settings
from storage import retention_store
from storage.audit_log import record_audit_event
from utils.helpers import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_started = False


def _run_cleanup_cycle() -> None:
    try:
        archived = retention_store.archive_expired()
    except Exception:  # noqa: BLE001 - a cleanup failure must never crash the app
        logger.exception("Document retention cleanup cycle failed")
        return

    for record in archived:
        record_audit_event("archival", record.document_id, {"archived": True})
    record_audit_event("cleanup_execution", "scheduler", {"archived_count": len(archived)})


def _loop() -> None:
    while True:
        _run_cleanup_cycle()
        time.sleep(max(1, settings.document_archive_cleanup_interval_seconds))


def start_retention_scheduler() -> None:
    """Start the background retention/archival thread, once per process.

    No-op if retention is disabled, or if the scheduler is already running.
    """
    global _started
    if not settings.document_retention_enabled:
        return
    with _lock:
        if _started:
            return
        thread = threading.Thread(target=_loop, name="document-retention-scheduler", daemon=True)
        thread.start()
        _started = True
        logger.info(
            "Document retention scheduler started (retention=%d min, interval=%ds)",
            settings.document_retention_minutes,
            settings.document_archive_cleanup_interval_seconds,
        )
