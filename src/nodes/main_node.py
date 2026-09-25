"""MainNode — orchestrator for the `main` backbone slot.

Runs the four middle sub-nodes sequentially, merging each result into a working
copy of state. Each sub-node is invoked as ``node(temp)`` so the framework-owned
``BaseNode.__call__`` runs the full 5-layer security pipeline (S-1 trust gate,
S-2 input gate, S-3 output gate, S-4 audit) for every inner node; MainNode also
emits a domain slot-level event.

Config delivery: the real SDK ``BaseNode.__call__(self, state)`` takes state only
and does NOT thread ``config`` into ``execute``. So MainNode resolves the manifest
config once and injects it into each config-dependent sub-node via its constructor
(the same injection pattern used for ``llm_client``). Passing ``config`` as a 2nd
positional arg to ``node(...)`` would raise ``TypeError`` in production.
"""

from typing import Any, ClassVar
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from src.nodes.api_name_detect_node import APINameDetectNode
from src.nodes.targeted_retrieve_node import TargetedRetrieveNode
from src.nodes.code_example_filter_node import CodeExampleFilterNode
from src.nodes.response_generate_node import ResponseGenerateNode
from src.nodes.config_resolver import resolve_agent_config
from src.utils.audit import emit_trace_event

_OUTPUT_KEYS = (
    "api_entities",
    "retrieved_chunks",
    "staleness_warning",
    "kb_unavailable",
    "code_examples",
    "raw_answer",
    "status",
)


class MainNode(FunctionNode):
    """Sequentially orchestrate APINameDetect -> TargetedRetrieve -> CodeExampleFilter -> ResponseGenerate."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm_client: Any = None, agent_cfg: dict[str, Any] | None = None) -> None:
        # SEC fix: carry the injected LLM client down to ResponseGenerateNode so the
        # graph wires a real shared.services.llm client end-to-end (no framework.llm).
        super().__init__()
        self._llm_client = llm_client
        # The manifest config, given at CONSTRUCTION. `execute(state)` receives no
        # config from the framework, so this is the only channel a caller has; `{}` is
        # an explicit "no settings", distinct from None meaning "read the file".
        self._agent_cfg = agent_cfg

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # CR-R3-01: resolve the manifest config once here and inject it into each
        # config-dependent sub-node's constructor, because __call__ (used below)
        # does not thread config into the sub-node's execute().
        # The INPUT was the problem: pre_process already wrote the sentence that says so.
        # Running the domain path anyway answers a question nobody asked -- "no matching
        # documentation found" is about a search, not about an empty message -- and the
        # shared envelope then prefers that sentence over the one the reader needs.
        if str(state.get("validation_error") or "").strip():
            return {"status": AgentStatus.SUCCESS.value}

        agent_cfg = self._agent_cfg if self._agent_cfg is not None else resolve_agent_config(None)
        temp = dict(state)
        for node in (
            APINameDetectNode(agent_cfg=agent_cfg),
            TargetedRetrieveNode(agent_cfg=agent_cfg),
            CodeExampleFilterNode(agent_cfg=agent_cfg),
            ResponseGenerateNode(llm_client=self._llm_client),
        ):
            # CR-R3-01: invoke through __call__ (single positional arg) so the
            # framework applies S-1/S-2/S-3/S-4 for every inner node. State carries
            # caller_trust_level (set by the framework at invoke time) for the S-1 gate.
            res = node(temp)
            if res.get("status") == AgentStatus.ERROR:
                emit_trace_event("main_error", {"error_node": node.__class__.__name__}, state)
                return res
            temp.update(res)
        emit_trace_event("main_complete", {"status": temp.get("status", "ok")}, state)
        return {k: temp[k] for k in _OUTPUT_KEYS if k in temp}
