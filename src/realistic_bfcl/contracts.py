from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallOracle:
    """Gold function call expected by the BFCL evaluator."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class BaseExample:
    """Clean BFCL example before any Realistic-BFCL transformation."""

    example_id: str
    prompt: str
    tool_schema: dict[str, Any]
    oracle: ToolCallOracle


@dataclass(frozen=True)
class NoisyExample:
    """Metamorphic variant derived from a clean BFCL example."""

    base_example_id: str
    dimension: str
    prompt_or_messages: str | list[dict[str, str]]
    tool_schema: dict[str, Any]
    oracle: ToolCallOracle
    audit_notes: list[str] = field(default_factory=list)


def check_oracle_preservation(base: BaseExample, noisy: NoisyExample) -> list[str]:
    """Return invariant violations for an oracle-preserving transformation."""

    violations: list[str] = []

    if noisy.base_example_id != base.example_id:
        violations.append("base_example_id does not match the clean example")
    if noisy.tool_schema != base.tool_schema:
        violations.append("tool_schema changed")
    if noisy.oracle.name != base.oracle.name:
        violations.append("oracle function name changed")
    if noisy.oracle.arguments != base.oracle.arguments:
        violations.append("oracle arguments changed")

    return violations
