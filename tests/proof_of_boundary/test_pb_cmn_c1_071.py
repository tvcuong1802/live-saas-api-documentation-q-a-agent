# CMN-C1-071 — Proof-of-Boundary tests PB-1, PB-2, PB-3, PB-6.
# (PB-4 import isolation = test_import_isolation.py; PB-5 state safety =
#  test_state_safety.py — both extended from the scaffold baseline.)
#
# S-4 trace is emitted via the module-level emit_trace_event imported into each
# node module (the real framework FunctionNode has NO _emit_trace_event hook).
# The trace spy therefore patches emit_trace_event in every node module.

import contextlib
from unittest.mock import patch

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from src.nodes.query_normalize_node import QueryNormalizeNode
from src.nodes.main_node import MainNode
from src.nodes.response_validate_node import ResponseValidateNode
from src.nodes.targeted_retrieve_node import TargetedRetrieveNode
from src.nodes.response_generate_node import ResponseGenerateNode

_NODE_MODULES = [
    "src.nodes.query_normalize_node",
    "src.nodes.api_name_detect_node",
    "src.nodes.targeted_retrieve_node",
    "src.nodes.code_example_filter_node",
    "src.nodes.response_generate_node",
    "src.nodes.response_validate_node",
]

_MOCK_CFG = {
    "agent": {
        "config": {
            "kb_backend": "mock",
            "configured_apis": [{"name": "kintone"}],
            "retrieval": {"top_k": 5, "score_threshold": -1.0},
        }
    }
}

_PRIMITIVES = (str, int, float, bool, type(None))


def _msgpack_safe(v) -> bool:
    if isinstance(v, _PRIMITIVES):
        return True
    if isinstance(v, list):
        return all(_msgpack_safe(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _msgpack_safe(x) for k, x in v.items())
    return False


@contextlib.contextmanager
def _trace_spy(events):
    """Patch emit_trace_event in every node module; record event_type names.

    Domain nodes emit DOMAIN events only — the framework backbone
    (BaseNode.__call__) owns node_start/node_complete/node_error.
    """

    def spy(event, payload=None, state=None):
        events.append(event)

    with contextlib.ExitStack() as stack:
        for mod in _NODE_MODULES:
            stack.enter_context(patch(f"{mod}.emit_trace_event", spy))
        yield


class TestPB1TraceEmitter:
    def test_domain_event_on_success(self):
        events = []
        with _trace_spy(events):
            QueryNormalizeNode().execute({"user_input": "hello"})
        assert "query_normalized" in events
        # backbone lifecycle events must NOT be emitted by the domain node
        assert "node_start" not in events
        assert "node_complete" not in events

    def test_domain_error_event_on_failure(self):
        events = []
        with _trace_spy(events):
            QueryNormalizeNode().execute({"user_input": "   ", "error_log": []})
        # Renamed with the contract: this path is not a failure, and an event called
        # "..._error" would keep saying it is to everyone reading the audit trail.
        assert "query_normalized_empty_input" in events
        assert "node_error" not in events


class TestPB2StateSerialization:
    def test_post_pipeline_state_is_msgpack_safe(self):
        s = {"user_input": "kintone GET /k/v1/records", "error_log": []}
        s.update(QueryNormalizeNode().execute(s))
        s.update(MainNode().execute(s))
        s.update(ResponseValidateNode().execute(s))
        assert all(_msgpack_safe(v) for v in s.values()), [k for k, v in s.items() if not _msgpack_safe(v)]
        # SEC-C1071-002: the structured fields are JSON strings, not nested dicts.
        for field in ("api_entities", "retrieved_chunks", "code_examples"):
            if field in s:
                assert isinstance(s[field], str)


class _FakeKB429:
    def search(self, *a, **k):
        raise RuntimeError("vector store error: HTTP 429 rate limited")


class TestPB3Fallback:
    def test_llm_failure_degrades_to_the_offline_answer(self):
        """A provider fault degrades; it does not fail the run.

        This asserted  and was named "degrades to error", which was never a
        degrade -- the runner raises on any status but SUCCESS, so the reader got
        "agent failed" while the offline answer this node can produce sat unused. What the
        test is really about is unchanged and still asserted: the node does not raise, the
        fault reaches the audit trail, and no  is emitted.
        """

        class _RaisingLLM:
            def generate(self, prompt):
                raise RuntimeError("timeout")

        events = []
        with _trace_spy(events):
            res = ResponseGenerateNode(llm_client=_RaisingLLM()).execute(
                {
                    "sanitized_query": "q",
                    "retrieved_chunks": '[{"endpoint": "/v1/auth", "method": "POST", "api_name": "Auth", "version": "v1", "chunk_text": "auth"}]',
                    "error_log": [],
                }
            )
        assert res["status"] == AgentStatus.SUCCESS.value
        # The offline answer is produced from the retrieved chunks, so the reader gets the
        # sources rather than a blank screen under "agent failed".
        assert res["raw_answer"].strip()
        # The reason is not swallowed: an operator looking for an outage still finds it.
        assert "response_model_unavailable" in events
        assert "node_error" not in events

    def test_vector_store_429_degrades_to_error(self):
        res = TargetedRetrieveNode(kb_service=_FakeKB429()).execute(
            {"sanitized_query": "q", "api_entities": "[]", "error_log": []}
        )
        assert res["status"] == AgentStatus.ERROR
        assert any("429" in e for e in res["error_log"])

    def test_backend_unavailable_fails_closed_to_safe_result(self):
        # KBBackendUnavailableError fails CLOSED to a safe empty result: status
        # success + kb_unavailable=True + zero chunks, so the pipeline continues to
        # a grounded no-docs answer instead of surfacing an error (STG-smoke runs
        # with no vector store provisioned and must still return status success).
        # A GENERIC search error (not backend-unavailable) still degrades to ERROR —
        # see test_vector_store_429_degrades_to_error above.
        # `agent_cfg={}` is the explicit "no settings" the test needs, given at
        # construction. It used to be a per-call `{"agent": {"config": {}}}`; with that
        # channel gone the node would otherwise read config.yaml and find a backend,
        # which is the opposite of the unavailable-backend case under test.
        res = TargetedRetrieveNode(agent_cfg={}).execute(
            {"sanitized_query": "q", "api_entities": "[]", "error_log": []},
        )
        assert res["status"] == AgentStatus.SUCCESS
        assert res["kb_unavailable"] is True
        assert res["retrieved_chunks"] == "[]"


class TestPB6LifecycleOrder:
    def test_six_node_order_via_manual_slot_chain(self):
        # ci_stub run()/_invoke_impl is a no-op (does not dispatch registered _nodes),
        # so the lifecycle is asserted by chaining the 3 slot nodes manually
        # (established another template pattern). MainNode emits no S-4 of its own.
        # Each domain node emits exactly one domain success event; the backbone
        # owns node_start/complete/error, so we assert the DOMAIN event order.
        _DOMAIN_SUCCESS = {
            "query_normalized",
            "api_detected",
            "kb_retrieved",
            "examples_filtered",
            "response_generated",
            "response_validated",
        }
        events = []
        with _trace_spy(events):
            s = {
                "user_input": "How to use kintone GET /k/v1/records",
                "error_log": [],
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
            s.update(QueryNormalizeNode().execute(s))
            s.update(MainNode().execute(s))
            s.update(ResponseValidateNode().execute(s))
        domain_order = [ev for ev in events if ev in _DOMAIN_SUCCESS]
        assert domain_order == [
            "query_normalized",
            "api_detected",
            "kb_retrieved",
            "examples_filtered",
            "response_generated",
            "response_validated",
        ]
