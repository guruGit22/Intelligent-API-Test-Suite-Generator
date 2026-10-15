"""Regression tests for contract-grounded test-case generation (offline
fallback and governance), using the fixture spec described in the mentor's
requirements (``samples/users_api.yaml``): no security schemes, no
minimum/maximum constraints, and no error response body schemas anywhere.

Covers acceptance criteria A-J, L, M from the "traceable to the uploaded
Swagger contract" requirement. Criterion K ($ref resolution / circular refs)
is covered in ``tests/test_contract_model.py``.
"""
from __future__ import annotations

import ast

from generators.postman_generator import generate_postman_environment, build_collection
from generators.pytest_generator import generate_pytest_suite
from llm.llm_generator import LLMGenerator, TestCase
from parser.openapi_parser import ParsedSpec, parse_spec


def _spec(samples_dir) -> ParsedSpec:
    return parse_spec(str(samples_dir / "users_api.yaml"))


def _endpoint(spec: ParsedSpec, method: str, path: str):
    return next(e for e in spec.endpoints if e.method == method and e.path == path)


def _fallback_cases(endpoint) -> list[TestCase]:
    # Calling generate_fallback_test_cases() directly never touches the OpenAI
    # client (self._client is never referenced) - proof that no network/API
    # call occurs, satisfying criterion L.
    generator = LLMGenerator(api_key="unused-for-this-test")
    return generator.generate_fallback_test_cases(endpoint)


def test_a_every_case_maps_to_a_real_operation(samples_dir):
    spec = _spec(samples_dir)
    valid_operations = {(e.operation_id, e.method, e.path) for e in spec.endpoints}
    for endpoint in spec.endpoints:
        for tc in _fallback_cases(endpoint):
            assert (tc.operation_id, tc.http_method, tc.endpoint_path) in valid_operations


def test_b_contract_defined_cases_use_documented_status_codes(samples_dir):
    spec = _spec(samples_dir)
    for endpoint in spec.endpoints:
        documented = {r.status_code for r in endpoint.responses}
        for tc in _fallback_cases(endpoint):
            if tc.is_contract_defined:
                assert str(tc.expected_status) in documented


def test_c_get_users_validates_array_of_user_schema(samples_dir):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "GET", "/users")
    cases = _fallback_cases(endpoint)

    happy = next(tc for tc in cases if tc.expected_status == 200)
    assert isinstance(happy.expected_response, list)
    assert any("array" in a.lower() for a in happy.assertions)
    # Item fields come from the User schema only - nothing invented.
    assert set(happy.expected_response[0].keys()) == {"id", "name", "email", "age", "status"}


def test_d_post_users_expects_201_not_generic_200(samples_dir):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "POST", "/users")
    cases = _fallback_cases(endpoint)

    happy_statuses = {tc.expected_status for tc in cases if tc.category == "Happy Path"}
    assert 201 in happy_statuses
    assert 200 not in happy_statuses


def test_e_post_users_schema_case_uses_create_user_request_required_fields(samples_dir):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "POST", "/users")
    cases = _fallback_cases(endpoint)

    schema_case = next(tc for tc in cases if tc.category == "Schema")
    joined = " ".join(schema_case.assertions)
    assert "name" in joined and "email" in joined and "age" in joined
    assert "status" not in joined  # not a CreateUserRequest field - never invented


def test_f_get_and_put_user_by_id_use_integer_path_parameter(samples_dir):
    spec = _spec(samples_dir)
    for method in ("GET", "PUT"):
        endpoint = _endpoint(spec, method, "/users/{userId}")
        path_params = [p for p in endpoint.parameters if p.location == "path"]
        assert len(path_params) == 1
        assert path_params[0].name == "userId"
        assert path_params[0].schema_type == "integer"
        assert path_params[0].required is True


def test_g_put_user_does_not_invent_required_fields(samples_dir):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "PUT", "/users/{userId}")
    cases = _fallback_cases(endpoint)

    # UpdateUserRequest documents no required fields -> no fabricated Schema case.
    assert not any(tc.category == "Schema" for tc in cases)


def test_h_delete_user_expects_204_without_body(samples_dir):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "DELETE", "/users/{userId}")
    cases = _fallback_cases(endpoint)

    case_204 = next(tc for tc in cases if tc.expected_status == 204)
    assert case_204.expected_response is None
    assert any("empty" in a.lower() for a in case_204.assertions)


def test_i_no_invented_auth_boundary_or_error_body_assertions(samples_dir):
    spec = _spec(samples_dir)
    for endpoint in spec.endpoints:
        cases = _fallback_cases(endpoint)
        # No security scheme is documented anywhere in this fixture.
        assert not any(tc.category == "Authorization" for tc in cases)
        # No min/max/length constraints are documented anywhere in this fixture.
        assert not any(tc.category == "Boundary" for tc in cases)
        # Undocumented-schema responses (400/404/409/500) must not carry body assertions.
        for tc in cases:
            response_match = next((r for r in endpoint.responses if r.status_code == str(tc.expected_status)), None)
            if response_match is not None and response_match.schema is None:
                assert tc.expected_response is None


def test_j_test_data_assumptions_are_labeled(samples_dir):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "GET", "/users/{userId}")
    cases = _fallback_cases(endpoint)

    not_found_case = next(tc for tc in cases if tc.expected_status == 404)
    assert not_found_case.is_contract_defined is True  # 404 itself IS documented
    assert any("test-data assumption" in note for note in not_found_case.assumptions)


def test_l_offline_fallback_never_calls_openai(samples_dir, monkeypatch):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "GET", "/users")
    generator = LLMGenerator(api_key="unused-for-this-test")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("Offline fallback must not call the OpenAI API")

    monkeypatch.setattr(generator, "_call_openai", _fail_if_called)
    cases = generator.generate_fallback_test_cases(endpoint)
    assert cases
    assert all(tc.generation_mode == "offline" for tc in cases)


def test_m_generators_still_work_with_contract_grounded_cases(samples_dir, tmp_path):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "POST", "/users")
    cases = _fallback_cases(endpoint)

    paths = generate_pytest_suite(endpoint, cases, "https://users.example.com/v1", output_dir=tmp_path)
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"))

    collection = build_collection("Users API", "https://users.example.com/v1", [(endpoint, cases)])
    assert collection["info"]["name"] == "Users API"

    environment_path = generate_postman_environment("Users API", "https://users.example.com/v1", tmp_path / "env.json")
    assert environment_path.exists()


def test_no_api_key_path_routes_generate_test_cases_to_offline_fallback(samples_dir):
    spec = _spec(samples_dir)
    endpoint = _endpoint(spec, "DELETE", "/users/{userId}")
    generator = LLMGenerator(api_key="")
    generator._has_api_key = False  # deterministic regardless of local .env

    cases = generator.generate_test_cases("unused prompt", endpoint=endpoint)
    assert generator.last_used_fallback is True
    assert cases
    assert all(tc.generation_mode == "offline" for tc in cases)
