"""ResponseValidateNode — post_process slot: strict citation grounding + confidence.

Grounding is endpoint-reference based ONLY (never chunk_text / source_path):
the cited (endpoint, method) must match a retrieved chunk's metadata.
"""

import re
from typing import Any, ClassVar
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.errors import SecurityViolationError
from src.nodes.config_resolver import resolve_agent_config
from src.nodes.response_generate_node import DETERMINISTIC_NOTICE_FORMS

from src.utils.audit import emit_trace_event
from src.utils.security import detect_credentials
from src.utils.serde import decode_list

# Texts this pipeline produces itself, never a model. Compared by exact match so a model
# that happens to phrase something similarly cannot claim the same free pass — and listing
# EVERY language form, because the set held only the bilingual ones for a while and the
# pipeline scored its own Japanese notice as ungrounded.
_DETERMINISTIC_NOTICES = DETERMINISTIC_NOTICE_FORMS

# Matches "/path METHOD" or "[endpoint: /path METHOD]", case-insensitive.
# Matches both "/path METHOD" / "[endpoint: /path METHOD]" and method-first
# "METHOD /path" (the form gpt-4o-mini often emits) — case-insensitive (F-2).
_CITATION_RE = re.compile(
    r"\[?\s*(?:endpoint:\s*)?(/[\w/{}.\-]+)\s+(GET|POST|PUT|PATCH|DELETE)\s*\]?"
    r"|"
    r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[\w/{}.\-]+)\b",
    re.IGNORECASE,
)
_DEFAULT_THRESHOLD = 0.7


class ResponseValidateNode(FunctionNode):
    """Validate citations are grounded in retrieved chunks; enforce confidence."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # The INPUT was the problem, and #pre_process already wrote the sentence that says
        # so. Composing a domain reply here puts a statement about a documentation search in
        # front of a reader who asked nothing, and the shared envelope prefers a prose
        # payload over `validation_error` -- so that sentence, not the refusal, is the one
        # they would read.
        if str(state.get("validation_error") or "").strip():
            emit_trace_event("response_validated", {"skipped": "validation_error"}, state)
            return {"status": AgentStatus.SUCCESS.value}
        try:
            # DR-071-4: honor an upstream ERROR — never mask it by resetting status.
            if state.get("status") == AgentStatus.ERROR:
                emit_trace_event("response_validated", {"upstream_error": True}, state)
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": state.get("error_log", []),
                    "final_answer": "",
                    "result": "",
                }
            raw_answer = state.get("raw_answer", "") or ""
            chunks = decode_list(state.get("retrieved_chunks"))
            agent_cfg = resolve_agent_config(None)
            threshold = float(agent_cfg.get("grounding_threshold", _DEFAULT_THRESHOLD))

            # Allowed endpoint references, from chunk METADATA only.
            ref_map: dict[tuple[str, str], str] = {}
            for c in chunks:
                ep = c.get("endpoint")
                method = c.get("method")
                if ep and method:
                    key = (str(ep).lower(), str(method).upper())
                    ref_map[key] = f"{c.get('api_name', '?')}@{c.get('version', '?')}{ep} {str(method).upper()}"

            # Distinct endpoints on both sides of the ratio. `total` used to count
            # citation OCCURRENCES while `validated` de-duplicated, so repeating a
            # correct citation lowered the score: one reference to /v1/auth scored
            # 1.0, two scored 0.5, three scored 0.33 — below the threshold, so the
            # answer that referenced its source most carefully was the one refused.
            # Measured on this node before the change.
            cited_keys: list[tuple[str, str]] = []
            for g in _CITATION_RE.findall(raw_answer):
                # g = (path1, method1, method2, path2); normalize either order.
                ep, method = (g[0], g[1]) if g[0] else (g[3], g[2])
                key = (ep.lower(), method.upper())
                if key not in cited_keys:
                    cited_keys.append(key)
            total = len(cited_keys)
            validated: list[str] = [ref_map[k] for k in cited_keys if k in ref_map]

            # DR-071-1: an answer that cites nothing while chunks were retrieved is
            # NOT grounded — no free pass. Full confidence only for a genuine
            # no-docs case (no chunks retrieved -> nothing to cite).
            if total:
                confidence = len(validated) / total
            elif not chunks:
                # No chunks and no citations describes two opposite situations, and
                # `total == 0` cannot tell them apart: the honest "no matching
                # documentation" answer, and a model answering from its own weights.
                # The first deserves full confidence; the second is ungrounded and used
                # to receive 1.0 as well, which is how an invented OAuth flow shipped
                # unchanged. Only the deterministic no-docs answer earns the free pass.
                # Both deterministic notices earn the pass: one says nothing matched,
                # the other says nothing was searched. Neither asserts anything about
                # an API, so neither can be ungrounded.
                confidence = 1.0 if raw_answer.strip() in _DETERMINISTIC_NOTICES else 0.0
            else:
                confidence = 0.0
            if confidence >= threshold:
                final_answer = raw_answer
            else:
                final_answer = (
                    "I cannot answer confidently — the available documentation does not "
                    "ground the requested API details."
                )

            # S-3 output gate (fail-closed): a credential / token must never leave the
            # pipeline, even if it survived the KB / LLM. Raises before the answer exits.
            # Scan the RAW answer, not just the substituted one: a low grounding score
            # replaces the text, and scanning only the replacement means a model that
            # leaked a token is reported as a routine low-confidence refusal. The
            # credential does not reach the user either way, but it stops being visible,
            # and a leaking model is exactly what this gate exists to surface.
            if detect_credentials(raw_answer) or detect_credentials(final_answer):
                raise SecurityViolationError("Credential detected in output.")

            emit_trace_event(
                "response_validated",
                {
                    "cited_total": total,
                    "validated_count": len(validated),
                    "grounding_confidence": confidence,
                    "passed_threshold": confidence >= threshold,
                },
                state,
            )
            return {
                "validated_citations": validated,
                "grounding_confidence": confidence,
                "final_answer": final_answer,
                "result": final_answer,
                "status": AgentStatus.SUCCESS.value,
            }
        except SecurityViolationError:
            emit_trace_event("response_validated_error", {}, state)
            raise
        except Exception as e:  # noqa: BLE001
            emit_trace_event("response_validated_error", {}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": state.get("error_log", []) + [f"ResponseValidate: {e}"],
            }
