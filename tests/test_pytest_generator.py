"""Tests for ``generators/pytest_generator.py``'s contract-validation and
request-payload enhancements. Existing parameter/header/status-code behavior
must keep working; new behavior must degrade cleanly when the spec documents
no schema (never inventing one).
"""
from __future__ import annotations

import ast

from generators.pytest_generator import generate_pytest_suite
from llm.llm_generator import TestCase as LLMTestCase
from parser.openapi_parser import parse_spec


def _endpoint(spec, method: str, path: str):
    return next(e for e in spec.endpoints if e.method == method and e.path == path)


def _test_case(expected_status: int, title: str = "case") -> LLMTestCase:
    return LLMTestCase(
        title=title,
        description="desc",
        priority="Medium",
        category="Happy Path",
        expected_status=expected_status,
        expected_response=None,
    )


def test_generated_file_is_syntactically_valid_with_body_and_schema(petstore_spec_path, tmp_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "POST", "/pets")

    paths = generate_pytest_suite(endpoint, [_test_case(201)], "https://api.example.com", output_dir=tmp_path)

    assert len(paths) == 1
    source = paths[0].read_text(encoding="utf-8")
    ast.parse(source)  # raises SyntaxError if the rendered file is invalid
    assert "json_body" in source
    assert "json=json_body" in source
    assert "jsonschema.validate" in source


def test_request_body_payload_uses_only_documented_fields(petstore_spec_path, tmp_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "POST", "/pets")  # body schema: Pet { id, name, tag }

    paths = generate_pytest_suite(endpoint, [_test_case(201)], "https://api.example.com", output_dir=tmp_path)
    source = paths[0].read_text(encoding="utf-8")

    assert "'id'" in source and "'name'" in source
    assert "invented_field" not in source


def test_undocumented_status_skips_schema_validation_but_stays_valid(petstore_spec_path, tmp_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "GET", "/pets/{petId}")

    paths = generate_pytest_suite(endpoint, [_test_case(418)], "https://api.example.com", output_dir=tmp_path)
    source = paths[0].read_text(encoding="utf-8")

    ast.parse(source)
    assert "jsonschema.validate" not in source


def test_existing_path_and_query_param_rendering_still_works(petstore_spec_path, tmp_path):
    spec = parse_spec(str(petstore_spec_path))
    endpoint = _endpoint(spec, "GET", "/pets/{petId}")

    paths = generate_pytest_suite(endpoint, [_test_case(200)], "https://api.example.com", output_dir=tmp_path)
    source = paths[0].read_text(encoding="utf-8")

    ast.parse(source)
    assert "path_params" in source
    assert 'url.replace("{" + name + "}"' in source
