"""Streamlit front end for the Intelligent API Test Suite Generator.

Two pages only:
1. Data Ingestion - upload a test-case dataset, route it into the knowledge
   base (test pattern library / security rule set / boundary value dataset /
   historical feedback store), then chunk & embed it into the FAISS vector index.
2. Test Case Generator - upload an OpenAPI/Swagger spec, see its endpoint
   classifications, then retrieve matching knowledge base entries per endpoint,
   build a prompt, call the LLM, and get formatted test cases back.

All business logic lives in the other modules (src/parser/, src/knowledge_base/,
src/llm/, src/validator/, src/generators/, src/storage/); this file only renders
the UI and manages Streamlit session state.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Code packages (ingestion/, parser/, generators/, etc.) now live under src/.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

from config import settings
from generators.postman_generator import generate_postman_collection, generate_postman_environment
from generators.pytest_generator import generate_pytest_suite
from knowledge_base.dataset_ingestion import ingest_dataset, knowledge_base_counts
from knowledge_base.vector_store import KnowledgeEntry, VectorStore, load_knowledge_entries
from llm.llm_generator import LLMGenerationError, LLMGenerator, TestCase
from llm.prompt_builder import build_feedback_notes, build_prompt
from parser.openapi_parser import Endpoint, OpenAPIParseError, ParsedSpec, parse_spec
from parser.schema_summary import summarize_endpoint
from storage.json_storage import storage
from storage import ingestion_draft_store, retention_store
from storage.audit_log import record_audit_event
from storage.retention_scheduler import start_retention_scheduler
from utils.helpers import get_logger
from validator.contract_validator import build_contract_check
from validator.coverage_validator import validate_coverage
from validator.duplicate_checker import deduplicate_test_cases
from ingestion.document_parser import parse_file, classify_document, load_ingestion_config
from ingestion.chunker import chunk_document
from ingestion.common_model import normalize_generic, normalize_openapi
from ingestion.mongo_document_tool import MongoDocumentRepository, DocumentUnderstandingTool
from ingestion.document_vector_store import DocumentVectorStore
from parser import contract_model
from ui_theme import (
    empty_state,
    format_file_size,
    format_timestamp,
    inject_global_styles,
    method_badge,
    render_app_header,
    render_page_header,
)

logger = get_logger(__name__)


def _init_session_state() -> None:
    defaults = {
        "parsed_spec": None,
        "spec_name": "",
        "spec_source": "",
        "test_cases_by_endpoint": {},
        "coverage_by_endpoint": {},
        "common_document_model": None,
        "base_url_override": storage.get_setting("base_url_override", ""),
        "feedback_status_by_key": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _endpoint_key(endpoint: Endpoint) -> str:
    return f"{endpoint.method} {endpoint.path}"


@st.cache_resource(show_spinner="Loading embedding model / vector index...")
def _get_vector_store() -> VectorStore:
    store = VectorStore()
    store.ensure_ready()
    return store


def _partition_entries(entries: list[KnowledgeEntry]) -> tuple[list[KnowledgeEntry], list[KnowledgeEntry], list[KnowledgeEntry]]:
    patterns = [e for e in entries if e.source == "test_patterns"]
    security = [e for e in entries if e.source == "security_rules"]
    boundary = [e for e in entries if e.source == "boundary_values"]
    return patterns, security, boundary


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

def render_dashboard() -> None:
    render_page_header(
        "Dashboard",
        "Overview of the knowledge base, ingested documents, and recent test generation activity.",
    )

    with st.container(key="dashboard_page"):
        kb_counts = knowledge_base_counts()
        col1, col2, col3, col4 = st.columns(4, gap="large")
        col1.metric("Test Patterns", kb_counts["test_pattern"])
        col2.metric("Security Rules", kb_counts["security_rule"])
        col3.metric("Boundary Values", kb_counts["boundary_value"])
        col4.metric("Historical Feedback", kb_counts["historical_feedback"])

        st.divider()

        col_spec, col_runs = st.columns(2, gap="large")
        with col_spec:
            st.markdown("#### Loaded specification")
            spec = st.session_state.get("parsed_spec")
            if spec is not None:
                with st.container(border=True, key="dashboard_spec_card"):
                    st.metric("Title", spec.title)
                    st.caption(f"Version {spec.version} · {len(spec.endpoints)} endpoint(s)")
            else:
                empty_state(
                    "🧪", "No specification loaded",
                    "Load an OpenAPI/Swagger spec from Test Case Generator to see it here.",
                )

        with col_runs:
            st.markdown("#### Recent test generation runs")
            recent_generated = storage.get_generated_tests(limit=5)
            if recent_generated:
                st.dataframe(
                    [
                        {
                            "Spec": r["spec_name"],
                            "Test cases": len(r["data"]) if isinstance(r["data"], list) else 0,
                            "Generated at": format_timestamp(r["timestamp"]),
                        }
                        for r in recent_generated
                    ],
                    use_container_width=True,
                    hide_index=True,
                    row_height=38,
                    column_config={
                        "Spec": st.column_config.TextColumn("Spec", width="large"),
                        "Test cases": st.column_config.NumberColumn("Test cases", width="small"),
                        "Generated at": st.column_config.TextColumn("Generated at", width="medium"),
                    },
                )
            else:
                empty_state(
                    "📄", "No test runs yet",
                    "Generate test cases from the Test Case Generator page to see them here.",
                )


# --------------------------------------------------------------------------- #
# Page 1: Data Ingestion
# --------------------------------------------------------------------------- #

def _clear_draft_query_param() -> None:
    if "draft_id" in st.query_params:
        del st.query_params["draft_id"]


def _find_existing_document(document_id: str) -> dict | None:
    """Best-effort duplicate check against MongoDB.

    Returns ``None`` (treated as "not a duplicate") if Mongo isn't configured
    or reachable rather than raising - this is an advisory check only and must
    never block the upload flow on its own.
    """
    try:
        repository = MongoDocumentRepository()
        repository.ping()
        return repository.get_document(document_id)
    except Exception:  # noqa: BLE001 - advisory check only
        return None


def render_data_ingestion() -> None:
    render_page_header(
        "Data Ingestion",
        "Upload backend and API documentation to build the knowledge and vector layer "
        "used for intelligent test generation.",
    )

    with st.container(key="ingestion_page"):
        cfg = load_ingestion_config()
        accepted = cfg.get("accepted_extensions", [])

        # The active draft_id lives in the URL query string (not session_state)
        # so it survives a full browser refresh, not just sidebar page switches.
        draft_id = st.query_params.get("draft_id", "")
        draft = ingestion_draft_store.get_draft(draft_id) if draft_id else None

        if draft is not None and draft.is_expired:
            if draft.status == "uploaded":
                st.warning("⚠ Upload expired, please upload again.")
            ingestion_draft_store.delete_draft(draft.draft_id)
            _clear_draft_query_param()
            draft = None

        file_bytes: bytes | None = None
        file_name: str | None = None
        file_type: str | None = None

        with st.container(border=True):
            st.markdown("#### Upload Backend Document")
            st.caption(
                "Upload Swagger, OpenAPI, JSON, YAML, XML, PDF, DOCX or other supported "
                "backend documentation."
            )

            if draft is not None and draft.status == "ingested":
                st.success(f"✓ `{draft.file_name}` was already uploaded and ingested successfully.")
                with st.container(border=True):
                    st.metric("Classification", draft.classification or "—")
                if st.button("Upload a different document"):
                    ingestion_draft_store.delete_draft(draft.draft_id)
                    _clear_draft_query_param()
                    st.rerun()
                return

            if draft is not None and draft.status == "uploaded":
                restored_bytes = ingestion_draft_store.read_draft_bytes(draft)
                if restored_bytes is None:
                    st.warning("⚠ The previously uploaded file could not be found. Please upload again.")
                    ingestion_draft_store.delete_draft(draft.draft_id)
                    _clear_draft_query_param()
                    draft = None
                else:
                    st.success(f"📄 Draft restored — `{draft.file_name}` is ready to ingest.")
                    file_bytes, file_name, file_type = restored_bytes, draft.file_name, draft.content_type
                    if st.button("Remove"):
                        ingestion_draft_store.delete_draft(draft.draft_id)
                        _clear_draft_query_param()
                        st.rerun()

            if draft is None:
                uploaded_file = st.file_uploader(
                    "Drag & drop your file here, or browse files",
                    type=[x.lstrip(".") for x in accepted],
                    help="Accepted formats are configuration-driven.",
                )
                st.caption(f"Supported formats: {', '.join(x.lstrip('.').upper() for x in accepted)}")

                if uploaded_file is not None:
                    import hashlib

                    candidate_bytes = uploaded_file.getvalue()
                    candidate_name = uploaded_file.name
                    candidate_type = uploaded_file.type or Path(uploaded_file.name).suffix
                    candidate_document_id = hashlib.sha256(candidate_bytes).hexdigest()[:24]

                    if _find_existing_document(candidate_document_id) is not None:
                        st.warning(
                            f"⚠ Duplicate document detected — `{candidate_name}` matches a document that was "
                            "already uploaded and ingested previously. Re-ingestion is disabled to avoid "
                            "duplicate knowledge-base entries."
                        )
                        try:
                            record_audit_event("upload", candidate_document_id, {"outcome": "duplicate_rejected"})
                        except Exception:  # noqa: BLE001 - audit logging must never block the UI
                            logger.exception("Audit logging failed for duplicate upload")
                        return

                    file_bytes = candidate_bytes
                    file_name = candidate_name
                    file_type = candidate_type

                    new_draft = ingestion_draft_store.create_draft(file_name, len(file_bytes), file_type, file_bytes)
                    st.query_params["draft_id"] = new_draft.draft_id
                    st.success(f"✓ File uploaded successfully — `{file_name}`")
                    try:
                        record_audit_event("upload", candidate_document_id, {"file_name": file_name})
                    except Exception:  # noqa: BLE001 - audit logging must never block the UI
                        logger.exception("Audit logging failed for upload")

        if file_bytes is None or file_name is None:
            empty_state(
                "📄", "No document uploaded",
                "Upload a backend/API document above to begin the ingestion pipeline.",
            )
            return

        if not st.button("Upload & Ingest", type="primary"):
            return

        active_draft_id = draft.draft_id if draft is not None else st.query_params.get("draft_id", "")


        import hashlib
        import tempfile

        document_id = hashlib.sha256(file_bytes).hexdigest()[:24]
        suffix = Path(file_name).suffix.lower()
        temp_path: Path | None = None

        status = st.status("Ingestion Pipeline", expanded=True)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                temp_path = Path(tmp.name)
            status.write("✅ Document uploaded")

            status.update(label="Parsing and classifying document...")
            raw_document = parse_file(str(temp_path))
            classification = classify_document(raw_document, suffix)
            status.write(f"✅ Document classified as `{classification}`")

            # The existing OpenAPI parser remains the authoritative parser for
            # API specifications; the result is normalized into the same common model.
            normalized = normalize_generic(
                raw_document,
                document_id=document_id,
                document_type=classification,
            )
            if classification == cfg.get("classification", {}).get("api_document_classification"):
                try:
                    parsed = parse_spec(str(temp_path))
                    normalized = normalize_openapi(parsed, document_id, classification)
                except OpenAPIParseError:
                    # Keep the generic normalized representation if the document
                    # was classified from content but is not a valid OpenAPI spec.
                    pass

            status.update(label="Connecting to MongoDB and storing document...")
            repository = MongoDocumentRepository()
            repository.ping()
            repository.save_document(
                document_id,
                file_name,
                classification,
                raw_document,
                normalized,
                classification,
            )
            status.write("✅ Stored in MongoDB")
            try:
                record_audit_event("ingestion", document_id, {"classification": classification})
            except Exception:  # noqa: BLE001 - audit logging must never block ingestion
                logger.exception("Audit logging failed for ingestion")

            status.update(label="Creating structure-aware chunks...")
            chunks = chunk_document(
                raw_document,
                document_id,
                max_chars=int(cfg.get("chunking", {}).get("max_characters", 1800)),
                overlap_chars=int(cfg.get("chunking", {}).get("overlap_characters", 200)),
            )
            repository.save_chunks(chunks)
            status.write(f"✅ Document chunked ({len(chunks)} chunks)")

            understanding_tool = DocumentUnderstandingTool(repository)
            understanding_result = understanding_tool.understand(document_id)
            status.write("✅ Structure analyzed")

            status.update(label="Embedding chunks and updating vector index...")
            vector_store = DocumentVectorStore()
            # MongoDB is the source of truth for the vector build.
            all_chunks = repository.get_all_chunks()
            vector_store.build(all_chunks)
            vector_store.save()
            status.write("✅ Embeddings generated")
            status.write("✅ Vector layer updated")

            try:
                retention_store.register_document(document_id, file_name, file_bytes)
                status.write("✅ Original document scheduled for retention & archival lifecycle")
            except Exception:  # noqa: BLE001 - retention bookkeeping must never block ingestion
                logger.exception("Document retention registration failed (non-blocking)")

            status.update(label="Ingestion complete", state="complete", expanded=False)

            if active_draft_id:
                # Keep the draft record (status "ingested", not deleted) and
                # its draft_id in the URL so returning to this page later
                # shows the document was already uploaded/ingested instead of
                # resetting to an empty upload state.
                ingestion_draft_store.mark_ingested(active_draft_id, document_id, classification)

            st.success(
                f"✓ Document ingested successfully — {len(chunks)} chunks were created "
                "and added to the vector layer."
            )

        except Exception as exc:  # noqa: BLE001 - UI boundary
            logger.exception("Data ingestion failed")
            status.update(label="Ingestion failed", state="error", expanded=True)
            if active_draft_id:
                # Per the persistence requirements, a failed upload must be
                # re-uploaded rather than silently resumed.
                ingestion_draft_store.delete_draft(active_draft_id)
                _clear_draft_query_param()
            st.error("❌ Unable to process document")
            st.caption(
                "The document could not be parsed, classified, or stored. Please verify "
                "the file format and backend connectivity, then try again."
            )
            with st.expander("Technical details"):
                st.code(str(exc))
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Page: Document Archive (retention/archival lifecycle status - standalone,
# does not alter the Data Ingestion or Test Case Generator pages/workflows)
# --------------------------------------------------------------------------- #

def render_document_archive() -> None:
    render_page_header(
        "Document Archive",
        "Retention and archival lifecycle status for originally uploaded documents. "
        "Metadata, embeddings, and test-case generation are unaffected by a document's "
        "archive status.",
    )

    with st.container(key="document_archive_page"):
        st.caption(
            f"Retention window: {settings.document_retention_minutes} minute(s) · "
            f"Retention enabled: {'Yes' if settings.document_retention_enabled else 'No'}"
        )

        records = retention_store.list_records()
        if not records:
            empty_state(
                "🗄", "No retained documents yet",
                "Documents are registered here automatically after a successful Data Ingestion run.",
            )
            return

        st.dataframe(
            [
                {
                    "Document": r.file_name,
                    "Status": r.status,
                    "Ingested at": format_timestamp(r.ingested_at),
                    "Retention until": format_timestamp(r.retention_until),
                    "Archived at": format_timestamp(r.archived_at) if r.archived_at else "—",
                    "Retrieved at": format_timestamp(r.retrieved_at) if r.retrieved_at else "—",
                }
                for r in records
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.markdown("#### Retrieve a document")
        st.caption("Retrieval goes through the retention store's authorized retrieval API only - no internal storage paths are ever shown.")
        options = {f"{r.file_name} ({r.status})": r.document_id for r in records if r.status != "Deleted"}
        if not options:
            st.caption("No retrievable documents (all have been deleted per retention policy).")
            return

        selected_label = st.selectbox("Document", list(options.keys()))
        if st.button("Retrieve"):
            document_id = options[selected_label]
            data = retention_store.retrieve_archived_document(document_id)
            if data is None:
                st.error("❌ Unable to retrieve this document (missing or already deleted).")
            else:
                record = retention_store.get_record(document_id)
                try:
                    record_audit_event("retrieval", document_id, {"status_after": record.status if record else None})
                except Exception:  # noqa: BLE001 - audit logging must never block retrieval
                    logger.exception("Audit logging failed for retrieval")
                st.success("✓ Document retrieved.")
                st.download_button(
                    "Download retrieved document", data,
                    file_name=record.file_name if record else "document",
                )


# --------------------------------------------------------------------------- #
# Page 2: Test Case Generator
# --------------------------------------------------------------------------- #

def _generate_tests_for_endpoint(endpoint: Endpoint, api_key: str) -> tuple[list[TestCase], object, bool]:
    """Run the retrieval + LLM (or offline fallback) generation pipeline for one
    endpoint. Returns (deduped test cases, coverage report, used_fallback)."""
    endpoint_summary = summarize_endpoint(endpoint)
    store = _get_vector_store()
    entries = store.query(endpoint_summary, top_k=10)
    patterns, security, boundary = _partition_entries(entries)
    feedback_notes = build_feedback_notes(storage.get_feedback())
    retrieved_document_chunks = []
    try:
        document_store = DocumentVectorStore()
        retrieved_document_chunks = document_store.query(endpoint_summary, top_k=5)
    except (FileNotFoundError, RuntimeError, ValueError):
        # Existing local knowledge-base retrieval remains available even when
        # the new Mongo/vector layer has not been configured yet.
        pass

    common_context = contract_model.build_operation_contract_dict(endpoint)
    prompt = build_prompt(
        endpoint_summary, patterns, security, boundary, feedback_notes,
        common_document_context=common_context,
        retrieved_document_chunks=retrieved_document_chunks,
    )

    generator = LLMGenerator(model=settings.openai_model, api_key=api_key)
    raw_cases = generator.generate_test_cases(prompt, endpoint=endpoint)
    deduped = deduplicate_test_cases(raw_cases)
    coverage = validate_coverage(endpoint, deduped)
    return deduped, coverage, generator.last_used_fallback


def render_test_case_generator() -> None:
    render_page_header(
        "Generate API Test Cases",
        "Upload an OpenAPI/Swagger specification and automatically generate comprehensive "
        "API test cases.",
    )

    with st.container(border=True):
        st.markdown("#### API Specification")
        url_col, upload_col = st.columns(2)
        with url_col:
            st.markdown("**Option 1 — Specification URL**")
            url = st.text_input(
                "Spec URL", placeholder="https://example.com/openapi.json",
                label_visibility="collapsed",
            )
        with upload_col:
            st.markdown("**Option 2 — Upload Specification**")
            generator_extensions = [
                x.lstrip(".") for x in load_ingestion_config().get("accepted_extensions", [])
            ]
            uploaded_file = st.file_uploader(
                "Drag & drop your Swagger/OpenAPI file here, or browse files",
                type=generator_extensions,
                key="spec_file_uploader",
                label_visibility="collapsed",
            )
        load_clicked = st.button("Load Specification", type="primary")

    if load_clicked:
        source: str | None = None
        temp_path: Path | None = None
        try:
            if url.strip():
                source = url.strip()
            elif uploaded_file is not None:
                suffix = Path(uploaded_file.name).suffix or ".json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = Path(tmp.name)
                source = str(temp_path)
            else:
                st.error("❌ Provide a specification URL or upload a file first.")
                return

            with st.spinner("Parsing and validating specification..."):
                spec = parse_spec(source)

            # Persist every frontend-uploaded specification so the document
            # understanding layer can consume a single MongoDB source of truth.
            try:
                import hashlib
                raw_for_mongo = parse_file(source)
                source_bytes = uploaded_file.getvalue() if uploaded_file is not None else source.encode("utf-8")
                document_id = hashlib.sha256(source_bytes).hexdigest()[:24]
                classification = classify_document(raw_for_mongo, Path(source).suffix.lower())
                repository = MongoDocumentRepository()
                normalized = normalize_openapi(spec, document_id, classification)
                repository.save_document(
                    document_id, Path(source).name, classification,
                    raw_for_mongo, normalized, classification
                )
                repository.save_chunks(
                    chunk_document(
                        raw_for_mongo, document_id,
                        max_chars=int(load_ingestion_config().get("chunking", {}).get("max_characters", 1800)),
                        overlap_chars=int(load_ingestion_config().get("chunking", {}).get("overlap_characters", 200)),
                    )
                )
                st.session_state["common_document_model"] = DocumentUnderstandingTool(repository).understand(document_id)
            except Exception as mongo_exc:  # Preserve the existing generator when Mongo is not configured.
                logger.warning("Mongo document persistence skipped: %s", mongo_exc)
                st.session_state["common_document_model"] = None

            st.session_state["parsed_spec"] = spec
            st.session_state["spec_name"] = spec.title
            st.session_state["spec_source"] = url.strip() or (uploaded_file.name if uploaded_file else "")
            st.session_state["test_cases_by_endpoint"] = {}
            st.session_state["coverage_by_endpoint"] = {}
            st.session_state["feedback_status_by_key"] = {}

            st.success(
                f"✓ Specification loaded successfully — **{spec.title}** (v{spec.version}), "
                f"{len(spec.endpoints)} endpoint(s)."
            )
        except OpenAPIParseError as exc:
            st.error("❌ Unable to load specification")
            st.caption(
                "The specification could not be parsed. Please verify the URL/file and try again."
            )
            with st.expander("Technical details"):
                st.code(str(exc))
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    spec: ParsedSpec | None = st.session_state["parsed_spec"]
    if spec is None:
        empty_state(
            "🧪", "No specification loaded",
            "Load an OpenAPI/Swagger spec above to see its endpoint classifications.",
        )
        return

    common_model = st.session_state.get("common_document_model")
    if common_model:
        common_model = common_model.get("common_parameters")
    else:
        common_model = normalize_openapi(
            spec,
            document_id=st.session_state.get("spec_name", "runtime"),
            document_type="api_specification",
        )
    with st.expander("Common document model — normalized parameters passed to the TC layer"):
        st.json(common_model)

    st.markdown("#### API Endpoints")
    rows = [
        {
            "Method": e.method,
            "Path": e.path,
            "Summary": e.summary or "",
            "Auth": ", ".join(a.type for a in e.auth) or "None",
            "Tags": ", ".join(e.tags),
        }
        for e in spec.endpoints
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander(f"Browse endpoint details ({len(spec.endpoints)})"):
        for e in spec.endpoints:
            method_col, path_col = st.columns([1, 6])
            with method_col:
                method_badge(e.method)
            with path_col:
                st.write(f"**{e.path}**  \n{e.summary or 'No summary provided.'}")
            with st.expander("Parameters / headers / request body / response / auth / schema", expanded=False):
                st.write("**Parameters**")
                st.json([
                    {
                        "name": p.name, "location": p.location, "required": p.required,
                        "type": p.schema_type, "default": p.default, "enum": p.enum,
                    }
                    for p in e.parameters
                ] or {"info": "No parameters documented for this operation"})
                st.write("**Headers**")
                st.json([
                    {"name": p.name, "required": p.required} for p in e.parameters if p.location == "header"
                ] or {"info": "No header parameters documented for this operation"})
                st.write("**Request body**")
                st.json(
                    e.request_body.schema
                    if e.request_body and e.request_body.schema
                    else {"info": "No request body documented for this operation"}
                )
                st.write("**Response**")
                st.json(
                    [{"status_code": r.status_code, "schema": r.schema} for r in e.responses]
                    or {"info": "No responses documented for this operation"}
                )
                st.write("**Authentication**")
                st.json([
                    {"name": a.name, "type": a.type, "scheme": a.scheme, "location": a.location}
                    for a in e.auth
                ] or {"info": "No authentication/security scheme documented for this operation"})
            st.divider()

    st.markdown("#### Generate Test Cases")
    options = [_endpoint_key(e) for e in spec.endpoints]
    selected = st.selectbox("Select an endpoint to inspect", options)
    endpoint = spec.endpoints[options.index(selected)]
    st.code(summarize_endpoint(endpoint), language="text")

    st.caption(f"{len(spec.endpoints)} endpoint(s) loaded from this specification.")
    col_single, col_all = st.columns(2)
    generate_single = col_single.button("Generate Test Cases for this endpoint", type="primary")
    generate_all = col_all.button(f"Generate Test Cases for ALL {len(spec.endpoints)} endpoints")

    if generate_single or generate_all:
        api_key = settings.openai_api_key
        if not api_key:
            st.info(
                "ℹ No OpenAI API key configured — generating offline, contract-grounded test cases "
                "directly from the uploaded specification instead of calling the LLM."
            )
        targets = spec.endpoints if generate_all else [endpoint]
        used_fallback_any = False
        total_generated = 0
        progress = st.progress(0.0) if generate_all else None
        try:
            for i, ep in enumerate(targets):
                with st.spinner(f"Generating test cases for {_endpoint_key(ep)}..."):
                    deduped, coverage, used_fallback = _generate_tests_for_endpoint(ep, api_key)

                ep_key = _endpoint_key(ep)
                st.session_state["test_cases_by_endpoint"][ep_key] = deduped
                st.session_state["coverage_by_endpoint"][ep_key] = coverage
                storage.save_generated_tests(st.session_state["spec_name"], deduped)
                storage.save_coverage(st.session_state["spec_name"], coverage)
                used_fallback_any = used_fallback_any or used_fallback
                total_generated += len(deduped)
                if progress is not None:
                    progress.progress((i + 1) / len(targets))

            if used_fallback_any:
                st.warning(
                    f"⚠ OpenAI API was not used (no key configured, quota exceeded, or rate-limited). "
                    f"Generated {total_generated} offline, contract-grounded test case(s) directly from "
                    "the uploaded specification for one or more endpoints."
                )
            else:
                st.success(f"✓ Generated {total_generated} test case(s) across {len(targets)} endpoint(s).")

            if generate_all:
                st.caption("Results per endpoint (switch the dropdown below to view each one's test cases):")
                st.dataframe(
                    [
                        {
                            "Endpoint": _endpoint_key(ep),
                            "Test cases": len(st.session_state["test_cases_by_endpoint"].get(_endpoint_key(ep), [])),
                        }
                        for ep in targets
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
        except LLMGenerationError as exc:
            st.error("❌ Test generation failed")
            st.caption("The LLM request could not be completed. Please try again.")
            with st.expander("Technical details"):
                st.code(str(exc))

    key = _endpoint_key(endpoint)
    test_cases: list[TestCase] = st.session_state["test_cases_by_endpoint"].get(key, [])
    coverage = st.session_state["coverage_by_endpoint"].get(key)

    if coverage:
        with st.expander("Coverage", expanded=True):
            st.code(coverage.as_text(), language="text")

    if test_cases:
        with st.expander("Contract Validation (against OpenAPI spec)", expanded=False):
            st.caption(
                "Checks each generated test case's expected status/response against what the "
                "original specification actually documents. Schema validation is skipped (not "
                "invented) when the spec does not define a schema for that status."
            )
            for i, tc in enumerate(test_cases):
                check = build_contract_check(endpoint, tc)
                st.code(f"TC-{i + 1:03d}\n{check.as_text()}", language="text")

    if test_cases:
        st.markdown(f"#### Generated Test Cases ({len(test_cases)})")
        st.caption(
            f"Showing test cases for the selected endpoint (**{key}**) only. "
            "Switch endpoints in the dropdown above to view the others - the Export step below always "
            "includes every endpoint with generated test cases."
        )
        st.dataframe(
            [
                {
                    "Test Case ID": tc.test_case_id or f"TC-{i + 1:03d}",
                    "Endpoint": endpoint.path,
                    "Method": endpoint.method,
                    "Operation": tc.operation_id or "—",
                    "Scenario": tc.category,
                    "Expected Result": tc.expected_status,
                    "Contract-defined": "Yes" if tc.is_contract_defined else "No (exploratory)",
                    "Status": st.session_state["feedback_status_by_key"].get(f"{key}_{i}", "Pending Review"),
                }
                for i, tc in enumerate(test_cases)
            ],
            use_container_width=True,
            hide_index=True,
        )

        for i, tc in enumerate(test_cases):
            with st.expander(f"TC-{i + 1:03d} · [{tc.category}] {tc.title} \u2014 expects {tc.expected_status}"):
                st.caption(
                    f"Operation: `{tc.operation_id or 'n/a'}` · "
                    f"{tc.http_method or endpoint.method} {tc.endpoint_path or endpoint.path}"
                )
                st.write(tc.description)
                st.write(f"**Priority:** {tc.priority}")
                if tc.assertions:
                    st.write("**Assertions:**")
                    for a in tc.assertions:
                        st.write(f"- {a}")
                with st.expander("Why this test? (contract source)"):
                    st.write(
                        "**Contract-defined:** Yes" if tc.is_contract_defined
                        else "**Contract-defined:** No - exploratory expectation, see assumptions below"
                    )
                    if tc.source_refs:
                        st.write("**Source reference(s):**")
                        for ref in tc.source_refs:
                            st.code(ref, language="text")
                    if tc.assumptions:
                        st.write("**Assumptions / limitations:**")
                        for note in tc.assumptions:
                            st.write(f"- {note}")
                col1, col2, col3, col4 = st.columns(4)
                if col1.button("\u2705 Approve", key=f"approve_{key}_{i}"):
                    storage.record_feedback("approved", tc)
                    st.session_state["feedback_status_by_key"][f"{key}_{i}"] = "Approved"
                    st.toast("Marked as approved (added to historical feedback store)")
                if col2.button("\u274C Reject", key=f"reject_{key}_{i}"):
                    storage.record_feedback("rejected", tc)
                    st.session_state["feedback_status_by_key"][f"{key}_{i}"] = "Rejected"
                    st.toast("Marked as rejected (added to historical feedback store)")
                note = col3.text_input("Modification note", key=f"note_{key}_{i}", label_visibility="collapsed", placeholder="What changed?")
                if col4.button("\U0001F4DD Modified", key=f"modified_{key}_{i}"):
                    storage.record_feedback("modified", tc, notes=note or None)
                    st.session_state["feedback_status_by_key"][f"{key}_{i}"] = "Modified"
                    st.toast("Marked as modified (added to historical feedback store)")

    test_cases_by_endpoint: dict[str, list[TestCase]] = st.session_state["test_cases_by_endpoint"]
    if not test_cases_by_endpoint:
        return

    st.markdown("#### Export")
    endpoint_by_key = {_endpoint_key(e): e for e in spec.endpoints}
    st.text_input(
        "Base URL (override)",
        value=st.session_state["base_url_override"] or (spec.base_url or ""),
        key="base_url_override",
        help="The URL exported pytest/Postman files will call, e.g. https://api.example.com.",
    )
    effective_base_url = st.session_state["base_url_override"].strip()
    if effective_base_url and effective_base_url != storage.get_setting("base_url_override"):
        storage.save_setting("base_url_override", effective_base_url)

    generate_environment = st.checkbox(
        "Also generate a matching Postman environment (base_url / auth_token variables)",
        value=True,
        help="auth_token is always left blank - fill it in locally in Postman, it is never exported.",
    )

    if st.button("Generate pytest + Postman files", type="primary", disabled=not effective_base_url):
        with st.spinner("Rendering pytest files and Postman collection..."):
            for ep_key, tcs in test_cases_by_endpoint.items():
                ep = endpoint_by_key.get(ep_key)
                if ep is None or not tcs:
                    continue
                generate_pytest_suite(ep, tcs, effective_base_url)

            endpoint_test_cases = [
                (endpoint_by_key[ep_key], tcs) for ep_key, tcs in test_cases_by_endpoint.items() if ep_key in endpoint_by_key
            ]
            postman_path = generate_postman_collection(spec.title, effective_base_url, endpoint_test_cases)
            environment_path = (
                generate_postman_environment(spec.title, effective_base_url) if generate_environment else None
            )
        st.success(
            f"✓ Generated pytest files in `{settings.generated_tests_dir}` and Postman "
            f"collection at `{settings.generated_postman_dir}`."
        )
        st.download_button("Download Postman collection", postman_path.read_bytes(), file_name=postman_path.name)
        if environment_path is not None:
            st.download_button(
                "Download Postman environment", environment_path.read_bytes(), file_name=environment_path.name
            )


def main() -> None:
    st.set_page_config(page_title="Intelligent API Test Suite Generator", layout="wide", page_icon="🧪")
    _init_session_state()
    start_retention_scheduler()
    inject_global_styles()
    render_app_header(
        "Intelligent API Test Suite Generator",
        "AI-powered API test generation and intelligent document ingestion",
    )

    pages = {
        "📄 Data Ingestion": render_data_ingestion,
        "🧪 Test Case Generator": render_test_case_generator,
        "🗄 Document Archive": render_document_archive,
    }
    with st.sidebar:
        st.markdown("### API Test Automation")
        selected_page = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

    pages[selected_page]()


if __name__ == "__main__":
    main()
