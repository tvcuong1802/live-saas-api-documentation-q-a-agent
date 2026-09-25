"""QueryNormalizeNode — pre_process slot: S-2 input gate + normalization.

- S-1 trust gate: ``required_trust_level`` (framework ``BaseNode.__call__`` enforces).
- S-2 input gate: prompt-injection + credential scan (fail-closed) before any retrieval.
- Language detection + query normalization.
"""

from typing import Any, ClassVar
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.errors import SecurityViolationError
from src.utils.audit import emit_trace_event
from src.services.agent_scope import INTAKE_POLICY
from src.services.input_intake import understand_input
from src.services.llm_provider import build_llm_client
from src.utils.security import detect_credentials, detect_injection


def _script_language(text: str) -> str | None:
    """Fallback only: what the characters are, when the model returns no usable code."""
    if any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in text or ""):
        return "ja"
    return "en" if text and text.strip() else None


class QueryNormalizeNode(FunctionNode):
    """S-2 input gate, then normalize the user query and detect its language."""

    # S-1 trust gate — enforced by the framework BaseNode.__call__ (ADR-006).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            user_input = state.get("user_input", "")
            if not user_input or not user_input.strip():
                # SUCCESS with a `validation_error`, not ERROR: an empty message is a
                # fact about what the sender sent, and they can act on it. The runner
                # raises on every status but SUCCESS, so reporting this as an error threw
                # away the guidance sentence written for precisely this case.
                # `validation_error` is the field the shared envelope reads; `error_log`
                # is where a crash leaves its trace, and mixing the two makes an unusable
                # input and a broken agent indistinguishable one layer later.
                emit_trace_event("query_normalized_empty_input", {}, state)
                return {
                    "status": AgentStatus.SUCCESS.value,
                    "validation_error": (
                        "There is no question here yet. Tell me which API or endpoint you "
                        "are asking about, and what you want to know about it.\n"
                        "ご質問の内容が読み取れませんでした。どの API・エンドポイントについて、"
                        "何をお知りになりたいかをご記載ください。"
                    ),
                }
            # S-2 input gate (fail-closed). Raises before any retrieval / LLM call.
            if detect_injection(user_input):
                raise SecurityViolationError("Prompt injection detected in input.")
            if detect_credentials(user_input):
                raise SecurityViolationError("Credentials detected in input.")
            normalized = " ".join(user_input.strip().split())
            # Which language to answer in. Reading the characters answers a different
            # question: a Japanese speaker writing romaji types only Latin, and was
            # answered in English (measured 2026-08-28). The intake call runs here --
            # inside execute(), therefore AFTER the S-2 gate above, so nothing that
            # failed the injection or credential scan is ever sent to the model.
            intake = understand_input(
                normalized,
                build_llm_client(state),
                policy=INTAKE_POLICY,
                script_language=_script_language,
            )
            language = intake["answer_language"]
            emit_trace_event(
                "query_normalized",
                {"language": language, "length": len(normalized), "language_source": intake["source"]},
                state,
            )
            return {
                "normalized_query": normalized,
                "query_language": language,
                "sanitized_query": normalized,
                "status": AgentStatus.SUCCESS.value,
            }
        except SecurityViolationError:
            emit_trace_event("query_normalized_error", {}, state)
            raise
        except Exception as e:  # noqa: BLE001 - boundary: convert to ERROR status
            emit_trace_event("query_normalized_error", {}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": state.get("error_log", []) + [f"QueryNormalize: {e}"],
            }
