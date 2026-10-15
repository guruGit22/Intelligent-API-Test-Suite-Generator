# Demo File Reference — Intelligent API Test Suite Generator

Standalone, click-through checklist of the files behind the **Data Ingestion** and
**Test Case Generator** modules, for use while presenting. This file is a
read-only reference only — it is not wired into the app's navigation, is not
imported anywhere, and does not affect any existing functionality.

> **How to use:** Click a link below to open that file in the editor (relative
> markdown links work both in the editor's Ctrl/Cmd+Click and in Markdown
> Preview). Paths with `#L<n>` jump straight to the relevant line/function.

---

## 📥 Data Ingestion

| Area | File | Notes |
|---|---|---|
| UI entry point | [app.py](../app.py#L184) | `render_data_ingestion()` |
| Ingestion config | [config/ingestion_config.json](../config/ingestion_config.json) | Accepted extensions, parser map, classification rules, chunking params |
| Upload draft persistence | [src/storage/ingestion_draft_store.py](../src/storage/ingestion_draft_store.py) | Survives page switches / browser refresh |
| Document parsing & classification | [src/ingestion/document_parser.py](../src/ingestion/document_parser.py) | `parse_file()`, `classify_document()` |
| Common document model | [src/ingestion/common_model.py](../src/ingestion/common_model.py) | `normalize_generic()`, `normalize_openapi()` |
| MongoDB storage | [src/ingestion/mongo_document_tool.py](../src/ingestion/mongo_document_tool.py) | `MongoDocumentRepository`, `DocumentUnderstandingTool` |
| Chunking | [src/ingestion/chunker.py](../src/ingestion/chunker.py) | `chunk_document()` |
| Embedding / FAISS index | [src/ingestion/document_vector_store.py](../src/ingestion/document_vector_store.py) | `DocumentVectorStore` |

---

## 📄 Sample Upload Files

| File | Notes |
|---|---|
| [data/samples/backend_document_for_ingestion.json](../data/samples/backend_document_for_ingestion.json) | Sample backend doc for the Data Ingestion upload |
| `samples/sample_openapi_for_test_case_generator.yaml` | ⚠ Not available — no file with this name exists under `data/samples/`. Use [data/samples/petstore.yaml](../data/samples/petstore.yaml) instead (listed below). |

---

## 🔑 Draft ID (`st.query_params`)

| Reference | File |
|---|---|
| Draft id read from URL query string | [app.py](../app.py#L197) |
| Draft id written to URL on upload | [app.py](../app.py#L305) |
| Draft id cleared from URL | [app.py](../app.py#L165) |
| Draft model & lifecycle (`uploaded` → `ingested` / `failed`) | [src/storage/ingestion_draft_store.py](../src/storage/ingestion_draft_store.py) |

---

## ⚙️ Test Case Generator

| Area | File | Notes |
|---|---|---|
| UI entry point | [app.py](../app.py#L472) | `render_test_case_generator()` |
| Sample spec | [data/samples/petstore.yaml](../data/samples/petstore.yaml) | OpenAPI/Swagger spec to upload |
| Sample dataset | [data/samples/sample_test_dataset.json](../data/samples/sample_test_dataset.json) | Existing sample test-case dataset |

---

*This reference was generated for demo purposes only. No application files, configuration, or data were modified, and the app was not run.*
