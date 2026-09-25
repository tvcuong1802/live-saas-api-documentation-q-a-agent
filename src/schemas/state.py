"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.

from typing import Optional
from framework.schemas.agent_state import AgentState


class State(AgentState, total=False):
    """CMN-C1-071 Live SaaS API Documentation Q&A Agent state.

    Flat, msgpack-safe TypedDict. Add only fields specific to this agent.
    Shared fields (user_input, status, session_id, node_history, error_log,
    hitl_*, etc.) are inherited from AgentState.

    Field groups follow the 6-node pipeline:
    QueryNormalize -> APINameDetect -> TargetedRetrieve -> CodeExampleFilter
    -> ResponseGenerate -> ResponseValidate.
    """

    # QueryNormalize (+ S-1 / S-2). No security_status field: S-1/S-2 are
    # fail-closed (raise SecurityViolationError); a pass leaves nothing for a
    # downstream node to read, so a status flag would be an orphan field.
    normalized_query: str
    query_language: str
    sanitized_query: str

    # APINameDetect — detected target API/endpoint entities.
    # SEC-C1071-002: stored as a JSON-serialized string (json.dumps of a
    # list[dict]) rather than list[dict[str, Any]] — `Any`-typed dicts are not
    # guaranteed msgpack-serializable and can corrupt the LangGraph checkpoint.
    # Decoded with json.loads() at the consuming node. Each decoded item:
    # {"api_name": str, "endpoint": str | None, "method": str | None}
    api_entities: Optional[str]

    # TargetedRetrieve — JSON-serialized list of chunks; each chunk carries
    # metadata {api_name, version, endpoint, method, last_updated} + text/score.
    retrieved_chunks: Optional[str]
    staleness_warning: str
    # Set True when the KB backend was unavailable at query time (fail-closed to an
    # empty, grounded no-docs answer rather than surfacing an error status).
    kb_unavailable: bool

    # CodeExampleFilter — JSON-serialized list of code-example chunks.
    code_examples: Optional[str]

    # ResponseGenerate
    raw_answer: str
    # True when retrieval returned nothing and the answer is the deterministic no-docs
    # reply, produced without calling the model. Declared so it survives the LangGraph
    # merge and shows up in the checkpoint: a run that answers nothing still reports
    # status success, and this flag is what tells the two apart after the fact.
    no_grounding: bool

    # ResponseValidate (+ S-3)
    validated_citations: list[str]
    grounding_confidence: float
    final_answer: str
    result: str
    #: The sentence saying the INPUT could not be used. Declared here because
    #: LangGraph merges only declared fields -- a node returned it, this class did not
    #: name it, and it was dropped before the envelope could turn it into a refusal.
    validation_error: str
