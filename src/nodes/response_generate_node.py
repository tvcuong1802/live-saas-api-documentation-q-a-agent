"""ResponseGenerateNode — LLM answer grounded in retrieved chunks, with citations."""

import re
from typing import Any, ClassVar
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.errors import SecurityViolationError
from src.services.llm_provider import build_llm_client
from src.utils.audit import emit_trace_event
from src.utils.serde import decode_list


# F-1: second-order injection guard — retrieved chunk text is untrusted (S-2 only
# scans user_input). Redact chunk text matching injection patterns before it reaches
# the LLM context, so a poisoned KB entry cannot hijack the prompt.
_CHUNK_INJECTION_RE = re.compile(
    # Tightened (F-1 follow-up): require an instruction-override qualifier so benign
    # API-doc text like "ignore deprecated fields ... validation rule" is NOT redacted.
    r"ignore\s+(?:all\s+|the\s+)?(?:previous|prior|above|preceding)\s+\w*\s*(?:instruction|rule|prompt|context|message)"
    r"|ignore\s+(?:all\s+)?(?:safety|security)\s+(?:guideline|instruction|rule|check|standard)"
    r"|disregard\s+(?:previous|all|above|prior)"
    r"|system\s+override|reveal\s+(?:the\s+)?(?:system\s+)?prompt"
    r"|you\s+are\s+now\s+(?:an?\s+)?(?:developer|assistant|unrestricted|jailbroken|free|different\s+ai|new\s+model|in\s+developer\s+mode)"
    r"|指示を無視|安全ガイドラインを無視|プロンプトを無視",
    re.IGNORECASE,
)


def _safe_chunk_text(text: str) -> str:
    return "[content removed: failed input-safety scan]" if _CHUNK_INJECTION_RE.search(text or "") else (text or "")


def _build_prompt(
    query: str, chunks: list[dict[str, Any]], code_examples: list[dict[str, Any]], language: str = "en"
) -> str:
    lines = [
        "You are a SaaS API documentation assistant. Answer ONLY from the provided context.",
        "Cite each fact with [endpoint: /path METHOD]. Do not invent endpoints.",
        f"Answer in the user's language ({language}).",
        "",
        f"Question: {query}",
        "",
        "Context:",
    ]
    for c in chunks:
        ref = f"{c.get('api_name', '?')}@{c.get('version', '?')} {c.get('endpoint', '')} {c.get('method', '')}".strip()
        lines.append(f"- [{ref}] {_safe_chunk_text(c.get('text', ''))}")
    if code_examples:
        lines.append("")
        lines.append("Code examples:")
        for c in code_examples:
            lines.append(f"- {_safe_chunk_text(c.get('text', ''))}")
    return "\n".join(lines)


# Bilingual when nothing decided a language, and the reader's language when something did.
#
# The old rule was "always both", on the argument that a query short enough to retrieve
# nothing is also where detection is least reliable. That argument holds for the case with
# no decision — which is still served both — but the agent DOES decide, per request, through
# the intake call in #pre_process, and ignoring a decision that was made is not caution. A
# question written in Japanese, long enough for the model to read it as Japanese, was still
# handed the English sentence first.
_NO_DOCS_EN = "No matching API documentation found for this query."
_NO_DOCS_JA = "この質問に一致する API ドキュメントは見つかりませんでした。"
_NO_DOCS_ANSWER = f"{_NO_DOCS_EN}\n{_NO_DOCS_JA}"


def _no_docs_answer(language: str) -> str:
    """The no-match sentence in the reader's language; both when the message named none."""
    code = str(language or "").strip().lower()
    if code == "ja":
        return _NO_DOCS_JA
    if code == "en":
        return _NO_DOCS_EN
    return _NO_DOCS_ANSWER


# The knowledge base being UNREACHABLE and the knowledge base containing NOTHING are
# different facts, and only one of them is about the user's question. Reporting the first
# as the second tells someone their API is undocumented when the truth is that nothing
# was looked up — a wrong answer delivered with status success.
_KB_UNAVAILABLE_EN = (
    "The documentation index could not be reached, so no search was performed. This is "
    "not a statement about whether your API is documented — please retry shortly, and "
    "report it if this persists."
)
_KB_UNAVAILABLE_JA = (
    "ドキュメント検索基盤に接続できなかったため、検索を実行していません。これは対象 API の"
    "ドキュメント有無を示すものではありません。しばらくしてから再度お試しください。"
)
_KB_UNAVAILABLE_ANSWER = f"{_KB_UNAVAILABLE_EN}\n{_KB_UNAVAILABLE_JA}"


def _kb_unavailable_answer(language: str) -> str:
    """The index-unreachable notice in the reader's language; both when none was decided."""
    code = str(language or "").strip().lower()
    if code == "ja":
        return _KB_UNAVAILABLE_JA
    if code == "en":
        return _KB_UNAVAILABLE_EN
    return _KB_UNAVAILABLE_ANSWER


#: EVERY form of the two notices this pipeline writes itself. The validator matches these
#: exactly to grant full confidence, so the moment a notice gained per-language forms, the
#: English-only and Japanese-only ones stopped being recognised — the pipeline scored its
#: own sentence as ungrounded and replaced it with "I cannot answer confidently".
DETERMINISTIC_NOTICE_FORMS = frozenset(
    {
        _NO_DOCS_EN,
        _NO_DOCS_JA,
        _NO_DOCS_ANSWER,
        _KB_UNAVAILABLE_EN,
        _KB_UNAVAILABLE_JA,
        _KB_UNAVAILABLE_ANSWER,
    }
)


def _offline_answer(chunks: list[dict[str, Any]], language: str = "") -> str:
    """What to say when documentation was found but no model is available to write it up.

    The previous text — "Based on the API documentation: [endpoint: ...]" — reads like
    an answer and is not one: it is a citation list with a sentence in front of it. On
    the platform this branch is reached when the LLM secrets are absent, so a user
    would receive that in place of a real answer while the run reported success. Saying
    plainly that the answer could not be generated, and still listing what matched, is
    both honest and more useful than the citation list alone.
    """
    cites = " ".join(
        f"[endpoint: {c.get('endpoint', '')} {c.get('method', '')}]".strip() for c in chunks if c.get("endpoint")
    )
    if not cites:
        return _no_docs_answer(language)
    return (
        "This answer could not be produced right now, so the sources below were not read "
        "and nothing here is an answer to the question. The following documentation "
        f"matches the query and can be consulted directly: {cites}\n"
        "現在この回答を生成できませんでした。以下の参照元は読まれておらず、質問への回答では"
        f"ありません。この質問に一致した参照元: {cites}"
    )


class ResponseGenerateNode(FunctionNode):
    """Generate the draft answer via the LLM."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm_client: Any = None) -> None:
        super().__init__()
        self._llm_client = llm_client

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            query = state.get("sanitized_query", "")
            chunks = decode_list(state.get("retrieved_chunks"))
            code_examples = decode_list(state.get("code_examples"))
            language = state.get("query_language", "en")
            # The DECISION as it stands, with no "en" default folded in: the prompt needs a
            # language to write in, but the no-docs sentence needs to know the difference
            # between "the reader wrote English" and "nobody decided" — the second is
            # answered in both, and defaulting it to English answers the wrong one.
            decided_language = str(state.get("query_language") or "")
            # SEC fix: LLM client is INJECTED (no non-public framework.llm import; it is
            # the stub harness-only → prod ImportError). When none is injected (offline / unit
            # tests), return a grounded mock answer citing the retrieved endpoints.
            if state.get("kb_unavailable"):
                # Say what actually happened. The retrieval node fails CLOSED to an empty
                # result so the pipeline keeps running, and that empty result is
                # indistinguishable from "nothing matched" by the time it arrives here —
                # the flag is the only thing that tells them apart.
                emit_trace_event("response_skipped_kb_unavailable", {"query_language": language}, state)
                return {
                    "raw_answer": _kb_unavailable_answer(decided_language),
                    "no_grounding": True,
                    "status": AgentStatus.SUCCESS.value,
                }
            if not chunks:
                # Nothing was retrieved, so there is nothing to ground an answer on.
                # Asking the model anyway is the classic RAG failure: it answers from
                # its own weights, cites nothing, and — because it cited nothing —
                # sailed through the citation check below with confidence 1.0.
                # Reproduced: an empty corpus produced a confident, entirely invented
                # OAuth flow for a SaaS API the agent had never read.
                #
                # Skipping the call is also the cheaper path: no tokens spent to
                # produce something that must then be thrown away.
                raw_answer = _no_docs_answer(decided_language)
                emit_trace_event("response_skipped_no_grounding", {"query_language": language}, state)
                return {"raw_answer": raw_answer, "no_grounding": True, "status": AgentStatus.SUCCESS.value}
            # The Marketplace runner builds the agent with no arguments, so the
            # `config={"llm": ...}` injection never happens there and this fell to the
            # offline branch below — shipping a stitched citation list as the answer
            # while still reporting success. Build from the invocation's own secrets
            # when nothing was injected. Kept LOCAL, never assigned to self: one node
            # instance serves concurrent invocations and each carries its own secrets.
            llm_client = self._llm_client or build_llm_client(state)
            if llm_client is not None:
                prompt = _build_prompt(query, chunks, code_examples, language)
                try:
                    raw_answer = llm_client.generate(prompt)
                except Exception as exc:  # noqa: BLE001 — a provider fault degrades, it does not fail
                    # A model call that fails at REQUEST time must degrade exactly like a
                    # model that was never reachable. This node already knows how to answer
                    # without one; what it did not do was take that path when the call
                    # itself failed, so a 401 or a provider outage fell through to
                    # . The runner raises on that, so the reader got
                    # "agent failed" instead of the offline answer sitting right here.
                    # The reason is not swallowed -- it goes to the audit trail, which is
                    # where an operator looks for an outage.
                    emit_trace_event("response_model_unavailable", {"reason": type(exc).__name__}, state)
                    llm_client = None
                    raw_answer = _offline_answer(chunks, decided_language)
            else:
                raw_answer = _offline_answer(chunks, decided_language)
            emit_trace_event(
                "response_generated",
                {"llm_used": llm_client is not None, "answer_length": len(raw_answer)},
                state,
            )
            return {"raw_answer": raw_answer, "status": AgentStatus.SUCCESS.value}
        except SecurityViolationError:
            emit_trace_event("response_generated_error", {}, state)
            raise
        except Exception as e:  # noqa: BLE001
            emit_trace_event("response_generated_error", {}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": state.get("error_log", []) + [f"ResponseGenerate: {e}"],
            }
