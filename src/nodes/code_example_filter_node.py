"""CodeExampleFilterNode — surface code-example chunks; drop stale-version chunks."""

from typing import Any, ClassVar
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.errors import SecurityViolationError
from src.nodes.config_resolver import resolve_agent_config
from src.utils.audit import emit_trace_event
from src.utils.serde import decode_list, encode_list


def _is_code_example(chunk: dict[str, Any]) -> bool:
    if chunk.get("chunk_type") == "code_example" or chunk.get("is_code"):
        return True
    text = str(chunk.get("text", ""))
    return "```" in text or "curl " in text or "import " in text


class CodeExampleFilterNode(FunctionNode):
    """Rank/keep relevant chunks and extract code examples."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, agent_cfg: dict[str, Any] | None = None) -> None:
        # CR-R3-01: manifest config injected by MainNode; falls back to the config
        # param for direct unit-test calls.
        super().__init__()
        self._agent_cfg = agent_cfg

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            chunks = decode_list(state.get("retrieved_chunks"))
            agent_cfg = self._agent_cfg if self._agent_cfg is not None else resolve_agent_config(None)
            include_code = agent_cfg.get("include_code_examples", True)

            code_examples = [c for c in chunks if _is_code_example(c)] if include_code else []
            emit_trace_event(
                "examples_filtered",
                {"code_example_count": len(code_examples), "include_code": bool(include_code)},
                state,
            )
            return {
                "code_examples": encode_list(code_examples),
                "retrieved_chunks": encode_list(chunks),
                "status": AgentStatus.SUCCESS.value,
            }
        except SecurityViolationError:
            emit_trace_event("examples_filtered_error", {}, state)
            raise
        except Exception as e:  # noqa: BLE001
            emit_trace_event("examples_filtered_error", {}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": state.get("error_log", []) + [f"CodeExampleFilter: {e}"],
            }
