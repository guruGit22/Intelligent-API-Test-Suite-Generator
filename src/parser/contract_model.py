"""The single, authoritative, normalized view of one OpenAPI operation.

Built on top of ``parser.openapi_parser.Endpoint`` (already fully ``$ref``-
resolved and validated by ``prance``/``openapi-spec-validator``) - this module
adds the schema-introspection, sample-generation, and traceability utilities
that both the online (LLM) and offline (fallback) test-case generators consume
as their *only* source of contract facts. Vector-search/RAG results are never
treated as contract facts here.

Nothing in this module invents a schema, required field, constraint, or status
code the specification does not define: absent facts are reported as absent
(``None``/empty list) or, for unresolved ``$ref``s, explicitly surfaced via
:func:`find_unresolved_refs` instead of being silently treated as valid.

No UI code lives here. Deliberately has no dependency on ``llm`` or
``validator`` packages so it can be imported from either without a circular
import.
"""
from __future__ import annotations

from typing import Any

from parser.openapi_parser import Endpoint, Parameter, ResponseDef

try:
    import jsonschema
except ImportError:  # pragma: no cover - jsonschema is declared in requirements.txt
    jsonschema = None


# --------------------------------------------------------------------------- #
# JSON-schema introspection (operates on already-$ref-resolved schema dicts)
# --------------------------------------------------------------------------- #

def find_unresolved_refs(schema: Any, _seen: frozenset[int] | None = None) -> list[str]:
    """Recursively scan a (supposedly fully resolved) schema for leftover
    ``$ref`` pointers, safely handling circular object references.

    ``prance`` resolves ``$ref``s while parsing the spec; any ``$ref`` still
    present in the result means it could not be resolved (missing/external
    reference, or a recursion cutoff for a circular schema) - this is reported
    here rather than ever being silently treated as a valid schema.
    """
    seen = _seen or frozenset()
    if isinstance(schema, dict):
        obj_id = id(schema)
        if obj_id in seen:
            return [f"<circular schema reference detected at {sorted(schema.keys())}>"]
        seen = seen | {obj_id}
        refs: list[str] = []
        if "$ref" in schema:
            refs.append(str(schema["$ref"]))
        for value in schema.values():
            refs.extend(find_unresolved_refs(value, seen))
        return refs
    if isinstance(schema, list):
        refs = []
        for item in schema:
            refs.extend(find_unresolved_refs(item, seen))
        return refs
    return []


def schema_required_fields(schema: dict[str, Any] | None) -> list[str]:
    """Documented ``required`` properties for an object schema (or an array's
    item schema). Never invents a required field the spec does not list."""
    if not schema:
        return []
    if schema.get("type") == "array":
        return schema_required_fields(schema.get("items"))
    return list(schema.get("required") or [])


_CONSTRAINT_KEYS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern", "minItems", "maxItems",
)


def schema_constraints(node: dict[str, Any]) -> dict[str, Any]:
    """Only the constraint keywords the schema node actually documents."""
    return {key: node[key] for key in _CONSTRAINT_KEYS if key in node}


def schema_properties(schema: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``name -> {type, format, enum, constraints, example}`` for an object
    schema (or an array's item schema). Only reports what the spec defines."""
    if not schema:
        return {}
    if schema.get("type") == "array":
        return schema_properties(schema.get("items"))
    properties = schema.get("properties") or {}
    result: dict[str, dict[str, Any]] = {}
    for name, node in properties.items():
        node = node or {}
        result[name] = {
            "type": node.get("type"),
            "format": node.get("format"),
            "enum": node.get("enum"),
            "constraints": schema_constraints(node),
            "example": node.get("example"),
        }
    return result


def is_array_schema(schema: dict[str, Any] | None) -> bool:
    return bool(schema) and schema.get("type") == "array"


def array_item_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if not schema:
        return None
    return schema.get("items")


_FORMAT_SAMPLES: dict[str, Any] = {
    "email": "user@example.com",
    "date": "2024-01-01",
    "date-time": "2024-01-01T00:00:00Z",
    "uuid": "00000000-0000-0000-0000-000000000000",
}


def sample_value_for_schema(schema: dict[str, Any] | None) -> object:
    """Best-effort sample value for one JSON-schema node.

    Only populates fields/values the schema actually defines (documented
    ``default``/``example``/``enum``/``format``) - never invents a new field
    name, enum member, or numeric bound.
    """
    schema = schema or {}
    if "default" in schema:
        return schema["default"]
    if "example" in schema:
        return schema["example"]
    if schema.get("enum"):
        return schema["enum"][0]
    schema_type = schema.get("type")
    schema_format = schema.get("format")
    if schema_format in _FORMAT_SAMPLES:
        return _FORMAT_SAMPLES[schema_format]
    if schema_type == "integer":
        return schema.get("minimum", 1)
    if schema_type == "number":
        return float(schema.get("minimum", 1))
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return [sample_value_for_schema(schema.get("items"))]
    if schema_type == "object" or schema.get("properties"):
        properties = schema.get("properties") or {}
        return {name: sample_value_for_schema(node) for name, node in properties.items()}
    return "sample"


def validate_response_against_schema(schema: dict[str, Any] | None, payload: Any) -> list[str]:
    """Validate ``payload`` against the documented JSON ``schema``.

    Returns a list of human-readable error messages (empty if valid or if there
    is nothing to validate). Reports — rather than silently skipping — when
    ``jsonschema`` is unavailable, and never raises on a malformed schema.
    """
    if schema is None:
        return []
    if jsonschema is None:
        return ["jsonschema is not installed; response schema validation was skipped"]
    try:
        validator = jsonschema.validators.validator_for(schema)(schema)
        return [
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in validator.iter_errors(payload)
        ]
    except Exception as exc:  # noqa: BLE001 - never let a malformed schema crash reporting
        return [f"Could not validate response against documented schema: {exc}"]


def find_response_definition(endpoint: Endpoint, status_code: int | str) -> ResponseDef | None:
    """Return the documented ``ResponseDef`` matching ``status_code``.

    Falls back to the spec's ``default`` response (if documented) when no exact
    status match exists. Returns ``None`` — never a fabricated definition — when
    the spec documents neither.
    """
    target = str(status_code)
    for response in endpoint.responses:
        if response.status_code == target:
            return response
    for response in endpoint.responses:
        if response.status_code.lower() == "default":
            return response
    return None


# --------------------------------------------------------------------------- #
# Traceability
# --------------------------------------------------------------------------- #

def _pointer_escape(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def operation_pointer(endpoint: Endpoint) -> str:
    """JSON-Pointer-style reference to this operation in the source document."""
    return f"#/paths/{_pointer_escape(endpoint.path)}/{endpoint.method.lower()}"


def response_source_ref(endpoint: Endpoint, status_code: str) -> str:
    return f"{operation_pointer(endpoint)}/responses/{status_code}"


# --------------------------------------------------------------------------- #
# Authoritative contract payload (fed into the LLM prompt / inspected by the UI)
# --------------------------------------------------------------------------- #

def _parameter_payload(param: Parameter) -> dict[str, Any]:
    return {
        "name": param.name,
        "location": param.location,
        "required": param.required,
        "type": param.schema_type,
        "format": param.format,
        "enum": param.enum,
        "default": param.default,
        "example": param.example,
        "constraints": param.constraints or None,
    }


def _response_payload(endpoint: Endpoint, response: ResponseDef) -> dict[str, Any]:
    schema = response.schema
    return {
        "status_code": response.status_code,
        "description": response.description,
        "content_type": response.content_type,
        "schema_available": schema is not None,
        "is_array": is_array_schema(schema),
        "required_fields": schema_required_fields(schema),
        "properties": schema_properties(schema),
        "unresolved_refs": find_unresolved_refs(schema),
        "source_ref": response_source_ref(endpoint, response.status_code),
    }


def build_operation_contract_dict(endpoint: Endpoint) -> dict[str, Any]:
    """Build the single authoritative, JSON-serializable contract payload for
    one operation. This is what gets fed into the LLM prompt (as the
    authoritative section) and consumed by the offline fallback generator -
    never raw vector-search chunks. Absent facts are omitted, not invented.
    """
    request_body = endpoint.request_body
    request_schema = request_body.schema if request_body else None

    return {
        "source_ref": operation_pointer(endpoint),
        "operation_id": endpoint.operation_id,
        "method": endpoint.method,
        "path": endpoint.path,
        "summary": endpoint.summary,
        "description": endpoint.description,
        "parameters": [_parameter_payload(p) for p in endpoint.parameters],
        "request_body": (
            {
                "required": request_body.required,
                "content_type": request_body.content_type,
                "required_fields": schema_required_fields(request_schema),
                "properties": schema_properties(request_schema),
                "unresolved_refs": find_unresolved_refs(request_schema),
            }
            if request_body is not None
            else None
        ),
        "responses": {r.status_code: _response_payload(endpoint, r) for r in endpoint.responses},
        "security": [
            {"name": a.name, "type": a.type, "scheme": a.scheme, "location": a.location}
            for a in endpoint.auth
        ],
    }
