"""Call the OpenAI GPT API to generate structured API test cases from a prompt
built by ``llm/prompt_builder.py``.

Parses/validates the JSON response into typed :class:`TestCase` objects, retrying
once with a "please return valid JSON only" repair prompt if the first response
fails to parse.

If no OpenAI API key is configured, or the API call fails because of a
quota/rate-limit/auth error (HTTP 429/401), falls back to offline test-case
generation. The offline fallback (see :func:`_contract_grounded_test_cases`)
consumes the *same* normalized OpenAPI contract (``parser.contract_model``) as
the online path - never generic, endpoint-agnostic templates - so the rest of
the pipeline (dedup, coverage, contract validation, generators, runners,
reports) is exercised with cases that are still traceable to the uploaded
specification even without a working OpenAI key.

Every returned test case (online or offline) is passed through
``_apply_contract_governance`` before being returned: operation/method/path are
stamped from the real endpoint (never trusted from the LLM), and any
expectation not actually documented by the spec is flagged
(``is_contract_defined=False``) with an explanatory assumption rather than
being silently dropped or replaced.

No UI code lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import APIError, OpenAI

from config import settings
from llm.prompt_builder import parse_llm_json_response
from parser.contract_model import (
    array_item_schema,
    find_response_definition,
    is_array_schema,
    operation_pointer,
    response_source_ref,
    sample_value_for_schema,
    schema_properties,
    schema_required_fields,
    validate_response_against_schema,
)
from parser.openapi_parser import Endpoint
from utils.helpers import get_logger

logger = get_logger(__name__)

VALID_CATEGORIES = {
    "Happy Path",
    "Negative",
    "Boundary",
    "Security",
    "Schema",
    "Authorization",
    "Edge",
}
VALID_PRIORITIES = {"High", "Medium", "Low"}


class LLMGenerationError(Exception):
    """Raised when the LLM cannot produce a usable, parseable set of test cases."""


_QUOTA_ERROR_CODES = {"insufficient_quota", "rate_limit_exceeded", "invalid_api_key"}


def _is_quota_or_rate_limit_error(exc: APIError) -> bool:
    """Return True if ``exc`` represents an HTTP 429 (quota/rate-limit) or 401
    (missing/invalid API key) error — cases where offline fallback should kick in.
    Never logs/exposes the API key itself.
    """
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)
    return code in _QUOTA_ERROR_CODES or status_code in (429, 401)


@dataclass
class TestCase:
    """A single generated API test case.

    The ``operation_id``/``http_method``/``endpoint_path``/``test_case_id``/
    ``source_refs``/``assumptions``/``is_contract_defined``/``generation_mode``
    fields are optional (default-populated) so existing callers/tests that
    construct a ``TestCase`` with only the original fields keep working; they
    are always filled in by ``_apply_contract_governance`` before a test case
    is returned from :meth:`LLMGenerator.generate_test_cases`.
    """

    title: str
    description: str
    priority: str
    category: str
    expected_status: int
    expected_response: Any
    assertions: list[str] = field(default_factory=list)
    operation_id: str | None = None
    http_method: str | None = None
    endpoint_path: str | None = None
    test_case_id: str = ""
    source_refs: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    is_contract_defined: bool = True
    generation_mode: str = "llm"


def _to_test_case(raw: dict[str, Any]) -> TestCase:
    priority = raw.get("priority", "Medium")
    if priority not in VALID_PRIORITIES:
        priority = "Medium"

    category = raw.get("category", "Edge")
    if category not in VALID_CATEGORIES:
        category = "Edge"

    try:
        expected_status = int(raw.get("expected_status", 0))
    except (TypeError, ValueError):
        expected_status = 0

    raw_assumptions = raw.get("assumptions") or []
    if not isinstance(raw_assumptions, list):
        raw_assumptions = [str(raw_assumptions)]

    return TestCase(
        title=str(raw.get("title", "")).strip() or "Untitled test case",
        description=str(raw.get("description", "")).strip(),
        priority=priority,
        category=category,
        expected_status=expected_status,
        expected_response=raw.get("expected_response"),
        assertions=[str(a) for a in raw.get("assertions", [])],
        assumptions=[str(a) for a in raw_assumptions],
        # is_contract_defined is re-derived authoritatively in
        # _apply_contract_governance - the LLM's own claim is not trusted.
    )


# --------------------------------------------------------------------------- #
# Offline fallback: contract-grounded test-case generation
# --------------------------------------------------------------------------- #

def _numeric_status(status_code: str) -> int | None:
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def _category_for_status(status: int) -> str:
    if 200 <= status < 300:
        return "Happy Path"
    if 400 <= status < 500:
        return "Negative"
    return "Edge"


def _property_assertions(schema: dict | None, prefix: str) -> list[str]:
    assertions = [f"{prefix} field '{name}' is present" for name in schema_required_fields(schema)]
    for name, info in schema_properties(schema).items():
        if info.get("type"):
            assertions.append(f"{prefix} field '{name}' has type '{info['type']}'")
        if info.get("format"):
            assertions.append(f"{prefix} field '{name}' matches documented format '{info['format']}'")
        if info.get("enum"):
            assertions.append(f"{prefix} field '{name}' is one of the documented values {info['enum']}")
    return assertions


def _response_test_case(endpoint: Endpoint, response, index: int) -> TestCase | None:
    """Build one contract-grounded test case for a single documented response.

    Returns ``None`` for non-numeric status codes (e.g. ``"default"``) - a
    concrete ``expected_status`` cannot be fabricated from those.
    """
    status = _numeric_status(response.status_code)
    if status is None:
        return None

    schema = response.schema
    category = _category_for_status(status)
    assumptions: list[str] = []

    if status == 204 or schema is None:
        expected_response = None
        assertions = [f"Response status code is {status}"]
        if status == 204:
            assertions.append("Response body is empty")
            assumptions.append("A 204 response has no body per the OpenAPI spec; only the status code is verified.")
        else:
            assumptions.append(
                f"No response schema is documented for status {status}; only the status code is verified."
            )
    elif is_array_schema(schema):
        item_schema = array_item_schema(schema)
        sample_item = sample_value_for_schema(item_schema)
        expected_response = [sample_item] if sample_item is not None else []
        assertions = ["Response body is a JSON array"] + _property_assertions(item_schema, prefix="Each array item")
    else:
        expected_response = sample_value_for_schema(schema)
        assertions = _property_assertions(schema, prefix="Response")

    if category == "Negative" and status == 404 and any(p.location == "path" for p in endpoint.parameters):
        id_param = next(p for p in endpoint.parameters if p.location == "path")
        assumptions.append(
            f"Uses an arbitrary '{id_param.name}' value as a test-data assumption; a documented 404 response "
            "does not prove that specific value does not exist in the target environment."
        )

    return TestCase(
        title=f"{endpoint.method} {endpoint.path} -> {status}",
        description=(
            f"Documented response for {endpoint.operation_id or endpoint.path} "
            f"({response.description or 'no description provided in the spec'})."
        ),
        priority="High" if category in ("Negative", "Security") else "Medium",
        category=category,
        expected_status=status,
        expected_response=expected_response,
        assertions=assertions,
        operation_id=endpoint.operation_id,
        http_method=endpoint.method,
        endpoint_path=endpoint.path,
        test_case_id=f"{endpoint.operation_id or endpoint.method}-{status}-{index}",
        source_refs=[response_source_ref(endpoint, response.status_code)],
        assumptions=assumptions,
        is_contract_defined=True,
        generation_mode="offline",
    )


def _request_body_schema_case(endpoint: Endpoint) -> TestCase | None:
    """A 'Schema' category case for the request body's documented required
    fields - only generated when the spec actually defines required fields."""
    request_body = endpoint.request_body
    if request_body is None or not request_body.schema:
        return None
    required_fields = schema_required_fields(request_body.schema)
    if not required_fields:
        return None

    documented_4xx = next(
        (r for r in endpoint.responses if (_numeric_status(r.status_code) or 0) // 100 == 4), None
    )
    if documented_4xx is not None:
        expected_status = int(documented_4xx.status_code)
        is_contract_defined = True
        assumptions: list[str] = []
    else:
        expected_status = 400
        is_contract_defined = False
        assumptions = ["400 is a conventional expectation for invalid input; not explicitly documented for this operation."]

    return TestCase(
        title=f"{endpoint.method} {endpoint.path}: missing required request field(s)",
        description=(
            f"Request body omits one of the documented required field(s) ({', '.join(required_fields)}) of "
            f"{endpoint.operation_id or endpoint.path}'s request schema."
        ),
        priority="High",
        category="Schema",
        expected_status=expected_status,
        expected_response=None,
        assertions=[f"Request omitting required field '{f}' is rejected" for f in required_fields],
        operation_id=endpoint.operation_id,
        http_method=endpoint.method,
        endpoint_path=endpoint.path,
        test_case_id=f"{endpoint.operation_id or endpoint.method}-schema-required",
        source_refs=[operation_pointer(endpoint) + "/requestBody"],
        assumptions=assumptions,
        is_contract_defined=is_contract_defined,
        generation_mode="offline",
    )


def _authorization_case(endpoint: Endpoint) -> TestCase | None:
    """An 'Authorization' case - only generated when the spec actually
    documents a security requirement for this operation."""
    if not endpoint.auth:
        return None
    documented_401 = next((r for r in endpoint.responses if r.status_code == "401"), None)
    is_contract_defined = documented_401 is not None
    assumptions = (
        []
        if is_contract_defined
        else ["401 is a conventional expectation for missing authentication; not explicitly documented for this operation."]
    )
    return TestCase(
        title=f"{endpoint.method} {endpoint.path}: missing/invalid authentication",
        description=(
            f"Call {endpoint.operation_id or endpoint.path} without valid credentials for its documented "
            "security requirement."
        ),
        priority="High",
        category="Authorization",
        expected_status=401,
        expected_response=None,
        assertions=["Request without valid credentials is rejected"],
        operation_id=endpoint.operation_id,
        http_method=endpoint.method,
        endpoint_path=endpoint.path,
        test_case_id=f"{endpoint.operation_id or endpoint.method}-auth",
        source_refs=[operation_pointer(endpoint) + "/security"],
        assumptions=assumptions,
        is_contract_defined=is_contract_defined,
        generation_mode="offline",
    )


def _boundary_cases(endpoint: Endpoint) -> list[TestCase]:
    """'Boundary' cases - only generated for parameters/request fields that
    actually document a constraint (minimum/maximum/length/etc). Skipped
    entirely (not invented) when no constraints are documented anywhere."""
    success_response = next(
        (r for r in endpoint.responses if (_numeric_status(r.status_code) or 0) // 100 == 2), None
    )
    if success_response is None:
        return []  # nothing safe to assert without a documented success status
    expected_status = int(success_response.status_code)

    candidates: list[tuple[str, dict, str]] = []
    for param in endpoint.parameters:
        if param.constraints:
            candidates.append((param.name, param.constraints, "parameter"))
    request_schema = endpoint.request_body.schema if endpoint.request_body else None
    for name, info in schema_properties(request_schema).items():
        if info.get("constraints"):
            candidates.append((name, info["constraints"], "request field"))

    cases: list[TestCase] = []
    for name, constraints, kind in candidates:
        assertions = [f"Documented {kind} '{name}' constraint {key} is {value}" for key, value in constraints.items()]
        cases.append(
            TestCase(
                title=f"{endpoint.method} {endpoint.path}: boundary values for '{name}'",
                description=f"Exercise the documented constraint(s) on {kind} '{name}': {constraints}.",
                priority="Medium",
                category="Boundary",
                expected_status=expected_status,
                expected_response=None,
                assertions=assertions,
                operation_id=endpoint.operation_id,
                http_method=endpoint.method,
                endpoint_path=endpoint.path,
                test_case_id=f"{endpoint.operation_id or endpoint.method}-boundary-{name}",
                source_refs=[operation_pointer(endpoint)],
                assumptions=[
                    f"Expected status {expected_status} assumes a boundary-valid '{name}'; the spec does not pin "
                    "a status specifically to this constraint test."
                ],
                is_contract_defined=False,
                generation_mode="offline",
            )
        )
    return cases


def _contract_grounded_test_cases(endpoint: Endpoint) -> list[TestCase]:
    """Build offline test cases from ``endpoint``'s normalized OpenAPI contract
    only (``parser.contract_model``) - never generic, endpoint-agnostic
    templates, and never a vector-search/knowledge-base chunk.
    """
    cases: list[TestCase] = []
    for i, response in enumerate(sorted(endpoint.responses, key=lambda r: r.status_code)):
        case = _response_test_case(endpoint, response, i)
        if case is not None:
            cases.append(case)

    schema_case = _request_body_schema_case(endpoint)
    if schema_case is not None:
        cases.append(schema_case)

    auth_case = _authorization_case(endpoint)
    if auth_case is not None:
        cases.append(auth_case)

    cases.extend(_boundary_cases(endpoint))

    if not cases:
        logger.warning(
            "No documented responses/contract facts available to ground offline fallback test cases for %s %s",
            endpoint.method, endpoint.path,
        )
    return cases


# --------------------------------------------------------------------------- #
# Contract governance: applied to every returned test case, online or offline
# --------------------------------------------------------------------------- #

def _apply_contract_governance(endpoint: Endpoint, test_cases: list[TestCase]) -> list[TestCase]:
    """Reconcile every test case against ``endpoint``'s real contract before it
    is ever shown in the UI or exported.

    - Stamps ``operation_id``/``http_method``/``endpoint_path`` from the real
      endpoint (never trusted from the LLM).
    - Flags (never silently drops) any expectation whose status code is not
      actually documented (``is_contract_defined=False`` + an explanatory
      assumption).
    - Repairs (clears) an ``expected_response`` that does not structurally
      match the documented schema, or that exists despite no schema being
      documented, recording why in ``assumptions`` rather than keeping an
      invented body.
    """
    for index, test_case in enumerate(test_cases):
        test_case.operation_id = endpoint.operation_id
        test_case.http_method = endpoint.method
        test_case.endpoint_path = endpoint.path

        target = str(test_case.expected_status)
        response_def = find_response_definition(endpoint, test_case.expected_status)
        documented = response_def is not None and response_def.status_code == target

        if not documented:
            test_case.is_contract_defined = False
            note = (
                f"Expected status {test_case.expected_status} is not documented in the OpenAPI spec for "
                f"{endpoint.method} {endpoint.path}; treated as an exploratory expectation."
            )
            if note not in test_case.assumptions:
                test_case.assumptions = [*test_case.assumptions, note]
        else:
            test_case.is_contract_defined = True
            schema = response_def.schema
            if schema is None:
                if test_case.expected_response not in (None, ""):
                    test_case.expected_response = None
                    test_case.assumptions = [
                        *test_case.assumptions,
                        f"No response schema is documented for status {test_case.expected_status}; "
                        "the expected response body was cleared rather than invented.",
                    ]
            elif isinstance(test_case.expected_response, (dict, list)):
                errors = validate_response_against_schema(schema, test_case.expected_response)
                if errors:
                    test_case.assumptions = [
                        *test_case.assumptions,
                        "Expected response body did not match the documented schema and was cleared: "
                        + "; ".join(errors),
                    ]
                    test_case.expected_response = None

        if not test_case.source_refs:
            test_case.source_refs = [
                response_source_ref(endpoint, target) if documented else operation_pointer(endpoint)
            ]
        if not test_case.test_case_id:
            test_case.test_case_id = f"{endpoint.operation_id or endpoint.method}-{test_case.category}-{index}"

    return test_cases


class LLMGenerator:
    """Thin wrapper around the OpenAI chat completions API for test-case generation."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self._model = model or settings.openai_model
        resolved_key = api_key or settings.openai_api_key
        # Never log/expose the key itself - only whether one is configured.
        self._has_api_key = bool(resolved_key)
        self._client = OpenAI(api_key=resolved_key or "unset")
        self.last_used_fallback = False

    def generate_fallback_test_cases(self, endpoint: Endpoint | None = None) -> list[TestCase]:
        """Build a contract-grounded test suite directly from ``endpoint``'s
        normalized OpenAPI contract, without calling the LLM.

        Used automatically by :meth:`generate_test_cases` when no API key is
        configured, or the OpenAI API returns a quota/rate-limit/auth error, so
        the rest of the pipeline (dedup, coverage, contract validation,
        generators, runners, reports) can still be exercised - with cases that
        remain traceable to the uploaded specification.
        """
        if endpoint is None:
            logger.warning(
                "Offline fallback requested without an endpoint - cannot ground test cases in a contract; "
                "returning no cases instead of generating generic/untraceable ones."
            )
            return []
        test_cases = _contract_grounded_test_cases(endpoint)
        logger.info(
            "Generated %d contract-grounded offline fallback test case(s) for %s %s",
            len(test_cases), endpoint.method, endpoint.path,
        )
        return test_cases

    def _call_openai(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You return JSON only, with no extra commentary."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMGenerationError("LLM returned an empty response")
        return content

    def _finalize(self, test_cases: list[TestCase], endpoint: Endpoint | None, mode: str) -> list[TestCase]:
        for test_case in test_cases:
            test_case.generation_mode = mode
        if endpoint is not None:
            test_cases = _apply_contract_governance(endpoint, test_cases)
        return test_cases

    def generate_test_cases(
        self,
        prompt: str,
        max_retries: int = 1,
        endpoint: Endpoint | None = None,
        allow_fallback: bool = True,
    ) -> list[TestCase]:
        """Send ``prompt`` to the LLM and return parsed, contract-governed test
        cases. Never assumes the OpenAI API is available: if no API key is
        configured, goes straight to the contract-grounded offline fallback.

        Args:
            prompt: The full prompt produced by ``llm.prompt_builder.build_prompt``.
            max_retries: How many times to retry with a JSON-repair instruction if
                parsing fails.
            endpoint: The endpoint being tested - the single source of truth used
                to ground both online and offline generation, and to govern
                (flag/repair) every returned test case.
            allow_fallback: If True (default), a 429/401 quota/rate-limit/auth
                error from OpenAI triggers offline, contract-grounded fallback
                generation instead of raising. Set to False to always raise on
                API errors.

        Raises:
            LLMGenerationError: if the API call fails (and fallback is disabled
                or not applicable), or the response still cannot be parsed after
                all retries are exhausted.
        """
        self.last_used_fallback = False

        if not self._has_api_key:
            logger.warning(
                "No OpenAI API key is configured - generating offline, contract-grounded fallback test cases."
            )
            self.last_used_fallback = True
            return self._finalize(self.generate_fallback_test_cases(endpoint), endpoint, mode="offline")

        current_prompt = prompt
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                raw_response = self._call_openai(current_prompt)
            except APIError as exc:
                logger.error("OpenAI API error on attempt %d: %s", attempt + 1, exc)
                if allow_fallback and _is_quota_or_rate_limit_error(exc):
                    logger.warning(
                        "OpenAI quota/rate-limit/auth error - falling back to offline, contract-grounded "
                        "test cases: %s", exc,
                    )
                    self.last_used_fallback = True
                    return self._finalize(self.generate_fallback_test_cases(endpoint), endpoint, mode="offline")
                raise LLMGenerationError(f"OpenAI API call failed: {exc}") from exc

            try:
                raw_cases = parse_llm_json_response(raw_response)
                test_cases = [_to_test_case(item) for item in raw_cases]
                logger.info("Generated %d test case(s) on attempt %d", len(test_cases), attempt + 1)
                return self._finalize(test_cases, endpoint, mode="llm")
            except ValueError as exc:
                last_error = exc
                logger.warning("Failed to parse LLM response on attempt %d: %s", attempt + 1, exc)
                current_prompt = (
                    f"{prompt}\n\nYour previous response was not valid JSON ({exc}). "
                    "Return ONLY a valid JSON array, with no markdown formatting or commentary."
                )

        raise LLMGenerationError(f"LLM did not return parseable JSON after {max_retries + 1} attempt(s): {last_error}")

