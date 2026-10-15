"""Central application configuration.

Loads configuration from environment variables (and a local ``.env`` file via
``python-dotenv``) and exposes a single immutable :class:`Settings` instance
(``settings``) that every other module should import instead of reading
``os.environ`` directly.

No UI code lives here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, as early as possible. Existing OS environment variables take
# precedence over values defined in the .env file.
load_dotenv(override=False)

# Root of the project (config.py now lives in config/, one level below root).
BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_path(name: str, default: str) -> Path:
    return (BASE_DIR / _env(name, default)).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable, process-wide application settings."""

    # --- MongoDB ---
    mongo_uri: str = field(default_factory=lambda: os.getenv("MONGO_URI", ""))
    mongo_database: str = field(default_factory=lambda: _env("MONGO_DATABASE", "api_test_generator"))
    mongo_document_collection: str = field(default_factory=lambda: _env("MONGO_DOCUMENT_COLLECTION", "documents"))
    mongo_chunk_collection: str = field(default_factory=lambda: _env("MONGO_CHUNK_COLLECTION", "document_chunks"))

    # --- Ingestion ---
    ingestion_config_file: Path = field(default_factory=lambda: _env_path("INGESTION_CONFIG_FILE", "config/ingestion_config.json"))

    # --- OpenAI ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))

    # --- FAISS / embeddings ---
    faiss_index_path: Path = field(
        default_factory=lambda: _env_path("FAISS_INDEX_PATH", "data/knowledge_base/faiss_index")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # --- Knowledge base ---
    knowledge_base_dir: Path = field(
        default_factory=lambda: _env_path("KNOWLEDGE_BASE_DIR", "data/knowledge_base")
    )
    test_patterns_file: Path = field(
        default_factory=lambda: _env_path("TEST_PATTERNS_FILE", "data/knowledge_base/test_patterns.json")
    )
    security_rules_file: Path = field(
        default_factory=lambda: _env_path("SECURITY_RULES_FILE", "data/knowledge_base/security_rules.json")
    )
    boundary_values_file: Path = field(
        default_factory=lambda: _env_path("BOUNDARY_VALUES_FILE", "data/knowledge_base/boundary_values.json")
    )
    historical_feedback_file: Path = field(
        default_factory=lambda: _env_path(
            "HISTORICAL_FEEDBACK_FILE", "data/knowledge_base/historical_feedback.json"
        )
    )

    # --- Generated artifacts ---
    generated_tests_dir: Path = field(
        default_factory=lambda: _env_path("GENERATED_TESTS_DIR", "tests/generated_tests")
    )
    generated_postman_dir: Path = field(
        default_factory=lambda: _env_path("GENERATED_POSTMAN_DIR", "artifacts/generated_postman")
    )

    # --- Storage (local JSON) ---
    storage_dir: Path = field(default_factory=lambda: _env_path("STORAGE_DIR", "src/storage/data"))

    # --- Data Ingestion drafts ---
    # Hours an in-progress (not-yet-ingested) Data Ingestion file upload stays
    # restorable after navigation/browser refresh before it is considered expired.
    ingestion_draft_ttl_hours: int = field(
        default_factory=lambda: int(_env("INGESTION_DRAFT_TTL_HOURS", "24"))
    )

    # --- Document retention & archival (Data Ingestion originals) ---
    document_retention_enabled: bool = field(
        default_factory=lambda: _env("DOCUMENT_RETENTION_ENABLED", "true").strip().lower() == "true"
    )
    document_retention_minutes: int = field(
        default_factory=lambda: int(_env("DOCUMENT_RETENTION_MINUTES", "15"))
    )
    document_archive_active_dir: Path = field(
        default_factory=lambda: _env_path("DOCUMENT_ARCHIVE_ACTIVE_DIR", "artifacts/archive/active")
    )
    document_archive_compressed_dir: Path = field(
        default_factory=lambda: _env_path("DOCUMENT_ARCHIVE_COMPRESSED_DIR", "artifacts/archive/compressed")
    )
    document_archive_cleanup_interval_seconds: int = field(
        default_factory=lambda: int(_env("DOCUMENT_ARCHIVE_CLEANUP_INTERVAL_SECONDS", "60"))
    )

    # --- Reports ---
    reports_dir: Path = field(default_factory=lambda: _env_path("REPORTS_DIR", "artifacts/reports/output"))

    # --- Templates ---
    templates_dir: Path = field(default_factory=lambda: BASE_DIR / "templates")

    # --- Logging ---
    logs_dir: Path = field(default_factory=lambda: _env_path("LOGS_DIR", "artifacts/logs"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- External tools ---
    newman_executable: str = field(default_factory=lambda: _env("NEWMAN_EXECUTABLE", "newman"))

    def ensure_directories(self) -> None:
        """Create all directories this app writes to, if they don't exist yet."""
        for directory in (
            self.knowledge_base_dir,
            self.generated_tests_dir,
            self.generated_postman_dir,
            self.storage_dir,
            self.storage_dir / "drafts",
            self.reports_dir,
            self.logs_dir,
            self.faiss_index_path.parent,
            self.document_archive_active_dir,
            self.document_archive_compressed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
