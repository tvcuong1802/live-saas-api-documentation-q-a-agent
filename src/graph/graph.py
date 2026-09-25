"""AgentCore Platform v1.0 — CMN-C1-071 graph (L1-direct, AgentBaseGraph).

Three-slot backbone (pre_process / main / post_process). The four middle
sub-nodes (APINameDetect → TargetedRetrieve → CodeExampleFilter → ResponseGenerate)
are orchestrated inside MainNode.

Security model (new-gen, real-SDK contract):
- S-1 trust gate: declared per node via ``required_trust_level`` and enforced by
  the framework ``BaseNode.__call__`` (ADR-006). No graph-level ``__pre_invoke__``.
- S-2 input injection/credential scan: inline in QueryNormalizeNode (pre_process).
- S-3 output credential-leak gate: inline in ResponseValidateNode (post_process).
- S-4 audit: ``src/utils/audit.emit_trace_event`` (module-level, real S-4 sink).

The old method-gate overrides (``__pre_invoke__`` / ``_security_gate_input`` /
``_security_gate_output`` / custom ``invoke``) were the stub harness-only and absent on the
real framework — removed. The injection/credential regexes live with the nodes.
"""

import re
from typing import Any
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from src.nodes.query_normalize_node import QueryNormalizeNode
from src.nodes.main_node import MainNode
from src.nodes.response_validate_node import ResponseValidateNode
from src.schemas.state import State
from src.utils.security import detect_credentials, detect_injection

# Selected by the reader's language, not stated in both. The comment this replaces said
# there is "no reliable language signal" on a rejected input because it is never
# normalised -- true of the normalised text, and the RAW message is still there.
# `_decided_language` reads its script, so a Japanese speaker whose request was refused
# is answered in Japanese instead of an English block with a translation underneath.
# Both halves still say what happened and what to do, and neither echoes the input or
# names what was detected in it.
_INPUT_REJECTED_BY_LANGUAGE = {
    "en": (
        "This request was not processed. The input was rejected by the agent's "
        "input-safety check, so no documentation was searched and no answer was "
        "generated. Please rephrase your question as a plain question about a SaaS "
        'API — for example: "How do I retrieve multiple records from a kintone app?"'
    ),
    "ja": (
        "このリクエストは処理されませんでした。入力が安全性チェックにより拒否されたため、"
        "ドキュメントの検索も回答の生成も行っていません。SaaS API に関する通常の質問の形に"
        "言い換えてください。例: 「kintone のレコードを取得する API の使い方は？」"
    ),
}

#: Both halves, for the one honest case: a message with no language to read.
_INPUT_REJECTED = _INPUT_REJECTED_BY_LANGUAGE["en"] + "\n" + _INPUT_REJECTED_BY_LANGUAGE["ja"]


def _input_rejected_text(state: Any) -> str:
    """The refusal in the reader's language, or both when there is none to read."""
    language = _decided_language(state)
    return _INPUT_REJECTED_BY_LANGUAGE.get(language or "", _INPUT_REJECTED)


class SaaSAPIDocQAAgent(AgentBaseGraph):
    """Live SaaS API Documentation Q&A Agent — inherits L1 AgentBaseGraph directly."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    @property
    def name(self) -> str:
        return "cmn_c1_071"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Register the 3 backbone slots. The 4 middle sub-nodes are orchestrated
        inside MainNode.execute() (same pattern as another template).

        The LLM client (if any) is injected via ``config["llm"]`` per the real-SDK
        contract; MainNode threads it down to ResponseGenerateNode.
        """
        super().register_nodes()
        cfg = self.config if isinstance(self.config, dict) else {}
        llm_client: Any = cfg.get("llm")
        self._nodes["pre_process"] = QueryNormalizeNode()
        self._nodes["main"] = MainNode(llm_client=llm_client)
        self._nodes["post_process"] = ResponseValidateNode()

    def get_output(self, state: dict[str, Any]) -> dict[str, Any]:
        """Public entry: delegate, then attach the trailer exactly once."""
        return _with_disclaimer(self._get_output_inner(state), state)

    def _get_output_inner(self, state: dict[str, Any]) -> dict[str, Any]:
        """Surface an S-2 input rejection as a readable refusal, not a blank error.

        The gate itself is unchanged: QueryNormalizeNode still raises
        SecurityViolationError, nothing is retrieved and no model is called. What
        changes is only what the caller reads. Before this, a rejected input came
        back as `{"output": null, "status": "error"}` — a blank screen with no
        reason — and the Marketplace runner turns any error status into
        "agent failed: RuntimeError: Graph invocation did not succeed", which is
        worse still. Measured on both paths.

        The rejection is identified by re-running THIS template's own S-2
        predicates on the input, not by matching the framework's error text: that
        text is a framework detail and would silently stop matching if it were
        reworded, taking the refusal message with it. Requiring the input to trip
        our own detector also means an unrelated failure cannot be relabelled — a
        rejected input never reaches retrieval, so no other error can be in play.
        """
        # Annotated because the wheel ships no py.typed, so super() resolves to Any
        # and mypy would otherwise flag every return from this method.
        result: dict[str, Any] = super().get_output(state)
        if result.get("status") != AgentStatus.ERROR.value or result.get("output"):
            return result
        user_input = state.get("user_input") or ""
        if not (detect_injection(user_input) or detect_credentials(user_input)):
            # A genuine failure — leave it alone. Masking an upstream error behind a
            # tidy message is how a broken agent comes to look like a working one.
            return result
        return {
            **result,
            "output": _input_rejected_text(state),
            "status": AgentStatus.SUCCESS.value,
        }


# Backwards-compat alias for direct imports / AgentRegistry.
Graph = SaaSAPIDocQAAgent


REFUSED_NOTICE = "This request was refused.\nこのリクエストは拒否されました。"


def _refused_before_answering(state: Any) -> bool:
    """True when the S-1 trust gate refused the caller, so there is no answer to dress.

    The refusal is stated and nothing else is: no scope line, no guidance. A caller who is
    not permitted to invoke the agent must not be told what it is for, and returning
    nothing is equally wrong -- the platform renders that as a blank screen under "agent
    failed". Read from `error_log`, which BaseNode.__call__ writes.
    """
    if not isinstance(state, dict):
        return False
    return any("S-1 trust gate denied" in str(e) for e in (state.get("error_log") or ()))


def _decided_language(state: Any) -> str | None:
    """The language this reader wants, or None when the message has none to read.

    `answer_language` is the decision the intake call makes from the message as received.
    Falling back to the script of the message covers the paths where S-2 rejected the
    input in the gate, so `execute()` never ran: the agent still knows WHO it is talking
    to even when it will not act on WHAT they said.

    Bilingual is now reserved for its one honest case -- punctuation, digits, an empty
    line. Passing None unconditionally, as this file used to, gave every reader whose
    request failed an English block with a Japanese translation stapled underneath.
    """
    if not isinstance(state, dict):
        return None
    code = str(state.get("answer_language") or "").strip().lower()[:2]
    if code in ("en", "ja"):
        return code
    for key in ("user_input", "raw_query", "validated_input", "normalized_query"):
        value = state.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        japanese = sum(1 for ch in value if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")
        # Latin WORDS, not characters: a Japanese message quoting an API key or a long id
        # used to read as English -- one 26-character token outweighed fourteen kana.
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", value) if 2 <= len(w) <= 20]
        latin = sum(len(w) for w in words)
        if japanese and japanese * 2 >= latin:
            return "ja"
        return "en" if latin else None
    return None


def _with_disclaimer(envelope: dict[str, Any], state: Any) -> dict[str, Any]:
    """Delegate to the shared envelope: one door for the whole fleet.

    This used to be a private copy. Keeping it private meant every change to the reply
    contract cost a refactor here instead of a file copy, and it silently missed the
    improvements the shared one gained -- the language decision, the single-closing-
    sentence rule, the blob check.

    Only the scope wording stays local, because only a reader of THIS agent can write it.
    """
    from src.services.agent_scope import SCOPE_EN, SCOPE_JA
    from src.services.output_envelope import with_disclaimer

    return with_disclaimer(envelope, state, scope_en=SCOPE_EN, scope_ja=SCOPE_JA)
