"""APINameDetectNode — extract target API / endpoint / method entities."""

import re
from typing import Any, ClassVar
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.errors import SecurityViolationError
from src.nodes.config_resolver import resolve_agent_config
from src.utils.audit import emit_trace_event
from src.utils.serde import encode_list

_ENDPOINT_RE = re.compile(r"(/[\w/{}.\-]+)")
_METHOD_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\b", re.IGNORECASE)


class APINameDetectNode(FunctionNode):
    """Detect which configured API(s)/endpoint(s) the query targets."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, agent_cfg: dict[str, Any] | None = None) -> None:
        # CR-R3-01: manifest config is injected by MainNode (constructor injection),
        # since __call__ does not thread config into execute(). Falls back to the
        # config param for direct unit-test calls.
        super().__init__()
        self._agent_cfg = agent_cfg

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            query = state.get("sanitized_query", "") or state.get("normalized_query", "")
            agent_cfg = self._agent_cfg if self._agent_cfg is not None else resolve_agent_config(None)
            configured = agent_cfg.get("configured_apis", []) or []
            known = [str(a.get("name", "")).lower() for a in configured if a.get("name")]

            ql = query.lower()
            ep_match = _ENDPOINT_RE.search(query)
            method_match = _METHOD_RE.search(query)
            endpoint = ep_match.group(1) if ep_match else None
            method = method_match.group(1).upper() if method_match else None

            # F-5 limitation: a single extracted endpoint/method is assigned to every
            # matched API. Fine for the single-API config today; revisit (per-API endpoint
            # extraction) before adding a second configured_api.
            entities: list[dict[str, Any]] = []
            for name in known:
                if name and name in ql:
                    entities.append({"api_name": name, "endpoint": endpoint, "method": method})

            emit_trace_event(
                "api_detected",
                {
                    "entity_count": len(entities),
                    "endpoint_found": endpoint is not None,
                    "method_found": method is not None,
                },
                state,
            )
            # SEC-C1071-002: persist as a JSON string (msgpack-safe checkpoint).
            return {"api_entities": encode_list(entities), "status": AgentStatus.SUCCESS.value}
        except SecurityViolationError:
            emit_trace_event("api_detected_error", {}, state)
            raise
        except Exception as e:  # noqa: BLE001
            emit_trace_event("api_detected_error", {}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": state.get("error_log", []) + [f"APINameDetect: {e}"],
            }
