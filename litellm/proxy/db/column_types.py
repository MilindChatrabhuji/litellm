from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import Final

_EMPTY_SCALAR_VALUES: Final[Mapping[str, str | int | None]] = MappingProxyType(
    {
        "request_id": "",
        "call_type": "",
        "api_key": "",
        "model": "",
        "spend": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "request_duration_ms": None,
        "model_id": None,
        "model_group": None,
        "custom_llm_provider": None,
        "api_base": None,
        "user": None,
        "cache_hit": None,
        "cache_key": None,
        "team_id": None,
        "organization_id": None,
        "end_user": None,
        "requester_ip_address": None,
        "session_id": None,
        "status": None,
        "mcp_namespaced_tool_name": None,
        "billing_agent_id": None,
        "agent_id": None,
        "litellm_call_id": None,
    }
)


def _needs_empty_value(column: str, value: object) -> bool:
    return isinstance(value, (list, tuple, set)) or (column == "agent_id" and isinstance(value, (int, float, Decimal)))


def conform_spend_log_columns(row: Mapping[str, object]) -> Mapping[str, object]:
    return {
        column: _EMPTY_SCALAR_VALUES[column]
        if column in _EMPTY_SCALAR_VALUES and _needs_empty_value(column, value)
        else value
        for column, value in row.items()
    }
