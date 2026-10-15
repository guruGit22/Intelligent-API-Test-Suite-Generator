"""Tests for ``validator/contract_validator.py`` - static OpenAPI contract
checks for generated test cases (status codes, response schemas, required
fields). Must never invent a schema the spec does not define.
"""
from __future__ import annotations

from llm.llm_generator import TestCase as LLMTestCase
from parser import contract_model
from parser.openapi_parser import parse_spec
from validator.contract_validator import build_contract_check, validate_response_against_schema


def _endpoint(spec, method: str, path: str):
    return next(e for e in spec.endpoints if e.method == method and e.path == path)


def _test_case(expected_status: int, expected_response=None) -> LLMTestCase:
    return LLMTestCase(
        title="case",
        description="desc",
        priority="Medium",
        category="Happy Path",
        expected_status=expected_status,
        expected_response=expected_response,
    )


def test_documented_status_with_valid_schema_has_no_errors(petstore_spec_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "GET", "/pets/{petId}")

    check = build_contract_check(endpoint, _test_case(200, {"id": 1, "name": "Rex"}))

    assert check.status_documented is True
    assert check.schema_available is True
    assert check.is_valid


def test_undocumented_status_is_reported_not_invented(petstore_spec_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "GET", "/pets/{petId}")

    check = build_contract_check(endpoint, _test_case(418))

    assert check.status_documented is False
    assert check.schema_available is False
    assert check.warnings  # explains why validation was skipped
    assert check.is_valid  # no fabricated errors either


def test_documented_status_without_schema_skips_validation(petstore_spec_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "POST", "/pets")

    check = build_contract_check(endpoint, _test_case(401))

    assert check.status_documented is True
    assert check.schema_available is False
    assert any("skipping" in w for w in check.warnings)


def test_expected_response_missing_required_field_is_flagged(petstore_spec_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "GET", "/pets/{petId}")

    check = build_contract_check(endpoint, _test_case(200, {"id": 1}))  # Pet requires id + name

    assert check.schema_available is True
    assert not check.is_valid
    assert any("name" in err for err in check.errors)


def test_validate_response_against_schema_handles_missing_jsonschema(monkeypatch):
    monkeypatch.setattr(contract_model, "jsonschema", None)
    errors = validate_response_against_schema({"type": "object"}, {"anything": True})
    assert errors == ["jsonschema is not installed; response schema validation was skipped"]


def test_validate_response_against_schema_returns_empty_when_no_schema():
    assert validate_response_against_schema(None, {"id": 1}) == []
