"""Merge an endpoint schema summary with retrieved knowledge-base context (test
patterns, OWASP security rules, boundary values) into a single structured prompt
for the LLM (see ``llm/llm_generator.py``).

No UI code lives here.
"""
from __future__ import annotations

import json

from knowledge_base.vector_store import KnowledgeEntry
from utils.helpers import get_logger

logger = get_logger(__name__)

RESPONSE_SCHEMA_INSTRUCTIONS = """\
Return JSON only. Do not include any explanation, markdown formatting, or text
outside the JSON array.

Return a JSON array where each element is a test case object with exactly these
fields:
- "title": string, short human-readable test name
- "description": string, what the test does and why
- "priority": one of "High", "Medium", "Low"
- "category": one of "Happy Path", "Negative", "Boundary", "Security", "Schema",
  "Authorization", "Edge"
- "expected_status": integer HTTP status code
- "expected_response": object or string describing the expected response shape
- "assertions": array of strings, each a concrete, checkable assertion
- "assumptions": array of strings - state anything about this test case that
  is NOT explicitly guaranteed by the OpenAPI Contract section below (e.g. an
  exploratory status code, an assumed test-data value, a conventional but
  undocumented expectation). Leave empty only if every expectation is directly
  documented in the contract.
"""

SYSTEM_INSTRUCTIONS = """\
You are an expert API test engineer. The "OpenAPI Contract" section below is
the single authoritative source of truth for this endpoint: its parameters,
request/response schemas, required fields, formats, constraints, documented
status codes, and security requirements. Only use facts that are explicitly
present in that section.

Treat "Relevant Test Patterns", "OWASP / Security Rules", "Boundary Values",
and "Retrieved Document Context" as supplementary guidance only - they may
suggest scenario *ideas*, but must never override, contradict, or extend
beyond what the OpenAPI Contract documents. Do not invent fields, response
properties, status codes, authentication requirements, or numeric/string
bounds that are absent from the contract. If a scenario requires an
undocumented expectation (e.g. an exploratory status code or an assumed
test-data value), include it but say so explicitly in "assumptions".

Generate a comprehensive set of API test cases for the given endpoint, using
the endpoint summary and OpenAPI Contract as ground truth. Cover as many of the
following categories as the contract actually supports: Happy Path, Negative,
Boundary, Security, Schema, Authorization, Edge. Do not fabricate a category
the contract gives no basis for (e.g. Authorization when no security scheme is
documented, or Boundary when no constraints are documented).
"""


def _format_entries(title: str, entries: list[KnowledgeEntry]) -> str:
    if not entries:
        return ""
    lines = [title]
    for entry in entries:
        lines.append(f"- [{entry.category}] {entry.title}: {entry.text}")
    return "\n".join(lines)


def build_prompt(
    endpoint_summary: str,
    test_patterns: list[KnowledgeEntry],
    security_rules: list[KnowledgeEntry],
    boundary_values: list[KnowledgeEntry],
    feedback_notes: list[str] | None = None,
    common_document_context: object | None = None,
    retrieved_document_chunks: list[dict] | None = None,
) -> str:
    """Build the full LLM prompt for one endpoint.

    Args:
        endpoint_summary: Output of ``parser.schema_summary.summarize_endpoint``.
        test_patterns: Retrieved test-pattern entries (from ``VectorStore.query``).
        security_rules: Retrieved OWASP/security-rule entries.
        boundary_values: Retrieved boundary-value entries.
        feedback_notes: Optional free-text notes distilled from
            ``data/knowledge_base/historical_feedback.json`` (e.g. previously
            QA-approved/rejected patterns) to steer generation.

    Returns:
        The complete prompt string to send to the LLM.
    """
    sections = [
        SYSTEM_INSTRUCTIONS,
        "Endpoint Summary\n----------------\n" + endpoint_summary,
    ]

    patterns_block = _format_entries("Relevant Test Patterns\n-----------------------", test_patterns)
    if patterns_block:
        sections.append(patterns_block)

    security_block = _format_entries("OWASP / Security Rules\n-----------------------", security_rules)
    if security_block:
        sections.append(security_block)

    boundary_block = _format_entries("Boundary Values\n---------------", boundary_values)
    if boundary_block:
        sections.append(boundary_block)

    if feedback_notes:
        sections.append(
            "Historical QA Feedback\n-----------------------\n" + "\n".join(f"- {note}" for note in feedback_notes)
        )

    if common_document_context:
        sections.append(
            "OpenAPI Contract (authoritative source of truth)\n"
            "-------------------------------------------------\n"
            + json.dumps(common_document_context, ensure_ascii=False, indent=2)
        )

    if retrieved_document_chunks:
        chunk_lines = ["Retrieved Document Context", "-------------------------"]
        for chunk in retrieved_document_chunks:
            chunk_lines.append(
                f"- path={chunk.get('path')} parent={chunk.get('parent_chunk_id')}\n"
                f"  {chunk.get('text', '')}"
            )
        sections.append("\n".join(chunk_lines))

    sections.append(RESPONSE_SCHEMA_INSTRUCTIONS)

    prompt = "\n\n".join(sections)
    logger.debug("Built prompt (%d chars)", len(prompt))
    return prompt


def build_feedback_notes(feedback_data: dict) -> list[str]:
    """Distill ``data/knowledge_base/historical_feedback.json`` into short guidance notes.

    Kept intentionally simple: summarizes counts and the most recent modified/
    rejected test titles so the prompt stays small.
    """
    notes: list[str] = []
    rejected = feedback_data.get("rejected", [])
    modified = feedback_data.get("modified", [])
    approved = feedback_data.get("approved", [])

    if approved:
        notes.append(f"{len(approved)} previously QA-approved test case(s) exist; favor similar style/coverage.")
    if rejected:
        titles = ", ".join(item.get("title", "unknown") for item in rejected[-5:])
        notes.append(f"Avoid patterns similar to previously rejected tests: {titles}")
    if modified:
        titles = ", ".join(item.get("title", "unknown") for item in modified[-5:])
        notes.append(f"QA previously modified tests like: {titles}. Prefer their corrected style.")
    return notes


def parse_llm_json_response(raw_response: str) -> list[dict]:
    """Parse the LLM's raw text response into a list of test-case dicts.

    Strips common wrapping artifacts (markdown code fences) before parsing.
    Raises ``ValueError`` if the content is not valid JSON or not a list.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("LLM response must be a JSON array of test case objects")

    return data
