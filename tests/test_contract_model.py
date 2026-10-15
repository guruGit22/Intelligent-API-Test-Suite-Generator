"""Tests for ``parser/contract_model.py`` - the single authoritative,
dependency-light contract/schema helpers shared by the online (LLM) and
offline (fallback) test-case generators. Covers $ref resolution via the real
parser, missing/unresolved references, arrays, nested schemas, and circular
references (acceptance criterion K).
"""
from __future__ import annotations

from parser import contract_model
from parser.openapi_parser import parse_spec


def _endpoint(spec, method: str, path: str):
    return next(e for e in spec.endpoints if e.method == method and e.path == path)


def test_find_unresolved_refs_reports_leftover_ref():
    schema = {"type": "object", "properties": {"owner": {"$ref": "#/components/schemas/Missing"}}}
    assert contract_model.find_unresolved_refs(schema) == ["#/components/schemas/Missing"]


def test_find_unresolved_refs_returns_empty_for_fully_resolved_schema():
    schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
    assert contract_model.find_unresolved_refs(schema) == []


def test_find_unresolved_refs_handles_circular_object_reference_safely():
    node_a: dict = {"type": "object", "properties": {}}
    node_b = {"type": "object", "properties": {"a": node_a}}
    node_a["properties"]["b"] = node_b  # circular: a -> b -> a

    # Must terminate (not recurse forever) and report the cycle rather than
    # silently treating it as an ordinary, fully-resolved schema.
    refs = contract_model.find_unresolved_refs(node_a)
    assert any("circular" in r for r in refs)


def test_find_unresolved_refs_handles_nested_arrays():
    schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"tag": {"$ref": "#/components/schemas/Missing"}}},
    }
    assert contract_model.find_unresolved_refs(schema) == ["#/components/schemas/Missing"]


def test_schema_required_fields_from_real_spec(samples_dir):
    spec = parse_spec(str(samples_dir / "users_api.yaml"))
    create_user = _endpoint(spec, "POST", "/users")
    assert contract_model.schema_required_fields(create_user.request_body.schema) == ["name", "email", "age"]


def test_schema_required_fields_empty_when_not_documented(samples_dir):
    spec = parse_spec(str(samples_dir / "users_api.yaml"))
    update_user = _endpoint(spec, "PUT", "/users/{userId}")
    assert contract_model.schema_required_fields(update_user.request_body.schema) == []


def test_schema_properties_reports_format(samples_dir):
    spec = parse_spec(str(samples_dir / "users_api.yaml"))
    create_user = _endpoint(spec, "POST", "/users")
    props = contract_model.schema_properties(create_user.request_body.schema)
    assert props["email"]["format"] == "email"
    assert props["age"]["type"] == "integer"


def test_is_array_schema_and_item_schema_cover_nested_refs(samples_dir):
    spec = parse_spec(str(samples_dir / "users_api.yaml"))
    get_users = _endpoint(spec, "GET", "/users")
    response_schema = next(r.schema for r in get_users.responses if r.status_code == "200")

    assert contract_model.is_array_schema(response_schema)
    item_schema = contract_model.array_item_schema(response_schema)
    assert set(contract_model.schema_properties(item_schema)) == {"id", "name", "email", "age", "status"}


def test_sample_value_for_schema_never_invents_fields():
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    sample = contract_model.sample_value_for_schema(schema)
    assert set(sample.keys()) == {"name", "age"}


def test_sample_value_for_schema_uses_format_aware_sample():
    assert contract_model.sample_value_for_schema({"type": "string", "format": "email"}) == "user@example.com"


def test_build_operation_contract_dict_has_no_unresolved_refs(samples_dir):
    spec = parse_spec(str(samples_dir / "users_api.yaml"))
    create_user = _endpoint(spec, "POST", "/users")
    contract = contract_model.build_operation_contract_dict(create_user)
    assert contract["request_body"]["unresolved_refs"] == []
    assert contract["operation_id"] == "createUser"
