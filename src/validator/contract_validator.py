"""Validate generated test-case expectations against the response contracts
documented in a parsed OpenAPI specification.

Builds on the pure, dependency-light schema/contract helpers in
``parser.contract_model`` (the single source of truth, shared with the online
and offline test-case generators in ``llm.llm_generator``) and adds the
``TestCase``-aware reporting layer used by the UI.

- ``build_contract_check``: a static, pre-execution check that compares a
  generated ``TestCase``'s expected status/response against what the spec
  actually documents for that endpoint (status codes, schemas, content types).
  Used to surface contract-validation results in the UI before any test runs.

Never fabricates a schema or expected status: when the specification does not
document a response for a given status code, or a documented response has no
schema, validation is skipped and reported as such instead of inventing one.

No UI code lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm.llm_generator import TestCase
from parser.contract_model import find_response_definition, validate_response_against_schema
from parser.openapi_parser import Endpoint
from utils.helpers import get_logger

logger = get_logger(__name__)

# Re-exported here for existing callers/tests - the implementations live in
# parser.contract_model (shared with llm.llm_generator, which cannot import
# this module without a circular import).
__all__ = [
    "ContractCheckResult",
    "find_response_definition",
    "validate_response_against_schema",
    "build_contract_check",
]


@dataclass
class ContractCheckResult:
    """Result of comparing one generated test case against the documented spec."""

    endpoint_path: str
    endpoint_method: str
    expected_status: int
    status_documented: bool
    schema_available: bool
    content_type: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_text(self) -> str:
        lines = [f"{self.endpoint_method} {self.endpoint_path} -> expected {self.expected_status}"]
        lines.append(
            "\u2713 Status is documented in the OpenAPI spec"
            if self.status_documented
            else "\u2717 Status is NOT documented in the OpenAPI spec"
        )
        if self.schema_available:
            lines.append(f"\u2713 Response schema available (content-type: {self.content_type or 'unspecified'})")
        else:
            lines.append("\u26a0 No response schema documented for this status - schema validation skipped")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        for error in self.errors:
            lines.append(f"Error: {error}")
        return "\n".join(lines)


def build_contract_check(endpoint: Endpoint, test_case: TestCase) -> ContractCheckResult:
    """Statically compare ``test_case``'s expectations against ``endpoint``'s
    documented responses, without executing any HTTP request.

    Args:
        endpoint: The endpoint the test case was generated for.
        test_case: A generated (LLM or offline-fallback) test case.

    Returns:
        A :class:`ContractCheckResult` describing what is documented and
        whether the test case's own expectations are consistent with it.
    """
    target = str(test_case.expected_status)
    response_def = find_response_definition(endpoint, test_case.expected_status)
    status_documented = response_def is not None and response_def.status_code == target
    schema = response_def.schema if response_def else None

    warnings: list[str] = []
    errors: list[str] = []

    if response_def is None:
        warnings.append(
            f"Expected status {test_case.expected_status} is not documented in the OpenAPI spec for "
            f"{endpoint.method} {endpoint.path}; contract validation will be skipped for this test case."
        )
    elif schema is None:
        warnings.append(
            f"No response schema is documented for status {test_case.expected_status}; skipping "
            "schema validation rather than inventing one."
        )
    elif isinstance(test_case.expected_response, (dict, list)):
        errors.extend(validate_response_against_schema(schema, test_case.expected_response))

    return ContractCheckResult(
        endpoint_path=endpoint.path,
        endpoint_method=endpoint.method,
        expected_status=test_case.expected_status,
        status_documented=status_documented,
        schema_available=schema is not None,
        content_type=response_def.content_type if response_def else None,
        errors=errors,
        warnings=warnings,
    )
