# CMN-C1-071 — Unit Tests: MainNode orchestrator (MR 3)

from src.nodes.main_node import MainNode
from src.utils.serde import decode_list
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel


def _mock_cfg():
    return {
        "agent": {
            "config": {
                "kb_backend": "mock",
                "configured_apis": [{"name": "kintone"}],
                "retrieval": {"top_k": 5, "score_threshold": -1.0},
            }
        }
    }


class TestMainNodeOrchestration:
    def test_happy_path_runs_all_subnodes(self):
        # CR-R3-01: MainNode.execute now invokes each sub-node via node(state), so the
        # real BaseNode.__call__ S-1 gate runs and reads caller_trust_level from state
        # (the framework sets it at invoke time). Provide it here so the gate passes.
        state = {
            "sanitized_query": "how to GET /k/v1/records in kintone",
            "error_log": [],
            "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        }
        res = MainNode(agent_cfg=_mock_cfg()["agent"]["config"]).execute(state)
        assert res["status"] == AgentStatus.SUCCESS
        # SEC-C1071-002: api_entities / retrieved_chunks are now JSON strings.
        assert any(e["api_name"] == "kintone" for e in decode_list(res["api_entities"]))
        assert decode_list(res["retrieved_chunks"]), "mock corpus should return kintone chunks"
        assert res["raw_answer"]

    def test_degrades_gracefully_when_backend_unavailable(self):
        # No kb_backend/docs/chroma -> TargetedRetrieve's KBService fails CLOSED to a
        # safe empty result (kb_unavailable=True). The orchestration continues to a
        # grounded no-docs answer and reports status success, NOT error (STG-smoke
        # contract: no vector store is provisioned during the smoke).
        state = {
            "sanitized_query": "kintone records",
            "api_entities": [],
            "error_log": [],
            "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        }
        # `agent_cfg={}` is the explicit "no settings" this case needs: no kb_backend,
        # no docs, no chroma. It was a per-call `{"agent": {"config": {}}}` before.
        res = MainNode(agent_cfg={}).execute(state)
        assert res["status"] == AgentStatus.SUCCESS
        assert res["kb_unavailable"] is True
        assert decode_list(res["retrieved_chunks"]) == []
        assert res["raw_answer"]
