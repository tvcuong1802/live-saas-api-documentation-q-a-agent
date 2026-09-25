"""TargetedRetrieveNode — API-scoped hybrid retrieval + staleness warning."""

from datetime import date, timedelta
from typing import Any, ClassVar
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.errors import SecurityViolationError
from src.nodes.config_resolver import resolve_agent_config
from src.services.kb_service import KBService, KBBackendUnavailableError
from src.utils.audit import emit_trace_event
from src.utils.serde import decode_list, encode_list

_DEFAULT_STALE_DAYS = 7


class TargetedRetrieveNode(FunctionNode):
    """Retrieve doc chunks scoped to the detected API(s)."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, kb_service: Any = None, agent_cfg: dict[str, Any] | None = None) -> None:
        # CR-R3-01: manifest config injected by MainNode; falls back to the config
        # param for direct unit-test calls.
        super().__init__()
        self._kb_service = kb_service
        self._agent_cfg = agent_cfg

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            query = state.get("sanitized_query", "")
            api_entities = decode_list(state.get("api_entities"))
            agent_cfg = self._agent_cfg if self._agent_cfg is not None else resolve_agent_config(None)
            retrieval = agent_cfg.get("retrieval", {}) if isinstance(agent_cfg, dict) else {}
            top_k = int(retrieval.get("top_k", 5))
            score_threshold = float(retrieval.get("score_threshold", 0.7))

            # Fail-closed to a SAFE empty result when the KB backend is unavailable
            # (e.g. chromadb not installed and no mock/docs backend — the STG smoke
            # runs with no vector store provisioned). This is NOT an error: the
            # pipeline continues with zero chunks, ResponseGenerate emits a grounded
            # "no matching documentation" answer, and the invoke returns status
            # success. A domain flag (kb_unavailable) records the degraded retrieval
            # so downstream / observability can distinguish it from a genuine
            # empty-result hit.
            try:
                kb = self._kb_service or KBService(config=agent_cfg)
                chunks = kb.search(query, api_entities, top_k=top_k, score_threshold=score_threshold)
            except KBBackendUnavailableError as ke:
                emit_trace_event("kb_unavailable", {"reason": str(ke)}, state)
                return {
                    "retrieved_chunks": encode_list([]),
                    "kb_unavailable": True,
                    "status": AgentStatus.SUCCESS.value,
                }

            result: dict[str, Any] = {
                "retrieved_chunks": encode_list(chunks),
                "status": AgentStatus.SUCCESS.value,
            }
            warning = self._staleness_warning(chunks, agent_cfg)
            if warning:
                result["staleness_warning"] = warning

            emit_trace_event("kb_retrieved", {"chunk_count": len(chunks), "stale": bool(warning)}, state)
            return result
        except SecurityViolationError:
            emit_trace_event("kb_retrieved_error", {}, state)
            raise
        except Exception as e:  # noqa: BLE001
            emit_trace_event("kb_retrieved_error", {}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": state.get("error_log", []) + [f"TargetedRetrieve: {e}"],
            }

    @staticmethod
    def _staleness_warning(chunks: list[dict[str, Any]], agent_cfg: dict[str, Any]) -> str:
        oldest = min((str(c.get("last_updated", "")) for c in chunks if c.get("last_updated")), default="")
        if not oldest:
            return ""
        retrieval = agent_cfg.get("retrieval", {}) if isinstance(agent_cfg, dict) else {}
        stale_after_days = int(
            agent_cfg.get("stale_after_days", retrieval.get("stale_after_days", _DEFAULT_STALE_DAYS))
        )
        cutoff = (date.today() - timedelta(days=stale_after_days)).isoformat()
        if oldest < cutoff:
            return f"Indexed documentation may be stale (oldest: {oldest}, threshold: {cutoff})."
        return ""
