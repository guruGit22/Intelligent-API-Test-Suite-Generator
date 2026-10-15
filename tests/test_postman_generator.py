"""Tests for ``generators/postman_generator.py`` - existing collection export
must keep working, and the new environment export must reference the same
variables (``base_url``, ``auth_token``) without hardcoding any secret value.
"""
from __future__ import annotations

from generators.postman_generator import build_collection, build_environment, generate_postman_environment
from llm.llm_generator import TestCase as LLMTestCase
from parser.openapi_parser import parse_spec


def _endpoint(spec, method: str, path: str):
    return next(e for e in spec.endpoints if e.method == method and e.path == path)


def _test_case() -> LLMTestCase:
    return LLMTestCase(
        title="Create a pet",
        description="desc",
        priority="Medium",
        category="Happy Path",
        expected_status=201,
        expected_response={"id": 1, "name": "Rex"},
    )


def test_existing_collection_export_still_works(petstore_spec_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "POST", "/pets")

    collection = build_collection("Petstore", "https://api.example.com", [(endpoint, [_test_case()])])

    assert collection["info"]["name"] == "Petstore"
    assert collection["variable"] == [
        {"key": "base_url", "value": "https://api.example.com"},
        {"key": "auth_token", "value": ""},
    ]
    assert collection["item"][0]["item"][0]["request"]["url"]["host"] == ["{{base_url}}"]


def test_environment_matches_collection_variable_names():
    environment = build_environment("Petstore", "https://api.example.com")

    keys = {v["key"] for v in environment["values"]}
    assert keys == {"base_url", "auth_token"}
    assert environment["_postman_variable_scope"] == "environment"


def test_environment_never_exports_a_real_token():
    environment = build_environment("Petstore", "https://api.example.com")

    auth_var = next(v for v in environment["values"] if v["key"] == "auth_token")
    assert auth_var["value"] == ""
    assert auth_var["type"] == "secret"


def test_environment_base_url_matches_requested_value():
    environment = build_environment("Petstore", "https://api.example.com")

    base_url_var = next(v for v in environment["values"] if v["key"] == "base_url")
    assert base_url_var["value"] == "https://api.example.com"


def test_generate_postman_environment_writes_valid_json(tmp_path):
    output_path = generate_postman_environment("Petstore", "https://api.example.com", tmp_path / "environment.json")

    assert output_path.exists()
    import json

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["name"] == "Petstore"
