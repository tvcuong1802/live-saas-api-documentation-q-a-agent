"""An answer with nothing behind it must not be presented as one with everything.

This is the RAG failure the whole template is built to avoid: retrieval returns
nothing, the model answers from its own weights, and — because it cited nothing —
the citation check has no citations to fault. Reproduced before the fix: an empty
corpus produced a confident, entirely invented OAuth flow at confidence 1.0.
"""

from __future__ import annotations

import json

import pytest

from src.nodes.response_generate_node import (
    _NO_DOCS_ANSWER,
    _NO_DOCS_EN,
    _NO_DOCS_JA,
    ResponseGenerateNode,
)
from src.nodes.response_validate_node import ResponseValidateNode

_CHUNK = {"endpoint": "/v1/auth", "method": "POST", "text": "POST /v1/auth returns a token."}


class _Hallucinating:
    """A model answering from its weights, which is what one does with no context."""

    called = 0

    def generate(self, prompt: str) -> str:
        type(self).called += 1
        return "POST your client_id and client_secret to /oauth2/token."


class _Citing:
    def generate(self, prompt: str) -> str:
        return "Use POST /v1/auth to obtain a token. [endpoint: /v1/auth POST]"


def _state(chunks: list[dict]) -> dict:
    return {
        "sanitized_query": "how do I authenticate?",
        "retrieved_chunks": json.dumps(chunks),
        "code_examples": json.dumps([]),
        "query_language": "en",
    }


def _run(chunks: list[dict], llm: object) -> dict:
    state = _state(chunks)
    state.update(ResponseGenerateNode(llm_client=llm).execute(state))
    return {**state, **ResponseValidateNode().execute(state)}


# ── the defect ────────────────────────────────────────────────────────────────


def test_empty_retrieval_never_reaches_the_model() -> None:
    """Nothing retrieved means nothing to ground on, so the model is not asked.

    Cheaper as well: no tokens spent producing an answer that has to be discarded.
    """
    _Hallucinating.called = 0
    out = _run([], _Hallucinating())

    assert _Hallucinating.called == 0, "the model was asked to answer with no context"
    # The fixture's reader wrote English, so the sentence is the English one. The
    # bilingual form is what a reader with NO decided language gets, and both forms are
    # pinned in `test_the_no_docs_sentence_follows_the_reader` below.
    assert out["raw_answer"] == _NO_DOCS_EN
    assert out["no_grounding"] is True


def test_empty_retrieval_answer_says_so_and_invents_nothing() -> None:
    out = _run([], _Hallucinating())
    assert out["final_answer"] == _NO_DOCS_EN
    assert "oauth2" not in out["final_answer"].lower(), "no endpoint the agent never read"


def test_no_docs_answer_keeps_full_confidence() -> None:
    """The honest refusal is a correct answer, not a failure.

    Both halves of the old branch matter: this one must stay 1.0 while the
    hallucinated one must not.
    """
    assert _run([], _Hallucinating())["grounding_confidence"] == 1.0


def test_uncited_answer_with_chunks_present_is_refused() -> None:
    """DR-071-1 preserved: chunks were retrieved and the answer cited none of them."""
    out = _run([_CHUNK], _Hallucinating())
    assert out["grounding_confidence"] == 0.0
    assert "cannot answer confidently" in out["final_answer"]


def test_cited_answer_passes() -> None:
    """The other direction — a grounded answer must still get through."""
    out = _run([_CHUNK], _Citing())
    assert out["grounding_confidence"] == 1.0
    assert "/v1/auth" in out["final_answer"]


# ── the validator on its own, without the generator in front of it ────────────


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # Every language form of the notice keeps full confidence. While the set held only
        # the bilingual one, the pipeline scored its own English and Japanese notices as
        # ungrounded and replaced them with "I cannot answer confidently".
        (_NO_DOCS_ANSWER, 1.0),
        (_NO_DOCS_EN, 1.0),
        (_NO_DOCS_JA, 1.0),
        (f"  {_NO_DOCS_ANSWER}  ", 1.0),
        ("POST to /oauth2/token to authenticate.", 0.0),
        ("", 0.0),
        ("No matching API documentation found for this query. Also, try /oauth2/token.", 0.0),
    ],
)
def test_validator_grants_full_confidence_only_to_the_exact_no_docs_answer(answer: str, expected: float) -> None:
    """Guards the check directly, in case a later change reintroduces an LLM call here.

    The last case is the one worth having: an answer that opens with the refusal and
    then volunteers an endpoint anyway is not a refusal.
    """
    state = {**_state([]), "raw_answer": answer}
    assert ResponseValidateNode().execute(state)["grounding_confidence"] == expected


# ── the ratio itself ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("See POST /v1/auth.", 1.0),
        ("Use POST /v1/auth. Details: POST /v1/auth.", 1.0),
        ("POST /v1/auth. Then POST /v1/auth. Finally POST /v1/auth.", 1.0),
        ("POST /v1/auth and POST /v1/secret.", 0.5),
        ("POST /v1/secret and GET /v1/admin.", 0.0),
    ],
)
def test_repeating_a_correct_citation_does_not_lower_the_score(answer: str, expected: float) -> None:
    """Distinct endpoints on both sides of the ratio, not occurrences against distinct.

    `total` counted citation occurrences while `validated` de-duplicated, so referencing
    the same endpoint twice scored 0.5 and three times 0.33 — under the 0.7 threshold,
    which meant the answer that pointed at its source most carefully was the one refused.
    Measured on this node before the change.

    The last two cases are what the ratio is actually for: one real endpoint and one
    invented gives 0.5, two invented gives 0.0.
    """
    state = {**_state([_CHUNK]), "raw_answer": answer}
    assert ResponseValidateNode().execute(state)["grounding_confidence"] == expected


def test_validated_citations_list_has_no_duplicates() -> None:
    """The reported citation list is a set of sources, not a tally of mentions.

    Separate from the ratio: with `validated` built from `cited_keys`, both sides of
    the fraction repeat together and the score stays right even without the
    de-duplication. What breaks is the list handed to the caller — an answer
    referencing one endpoint three times would report it three times. Found by
    mutation-testing: removing the dedup left every scoring test green.
    """
    state = {**_state([_CHUNK]), "raw_answer": "POST /v1/auth. Again POST /v1/auth. And POST /v1/auth."}
    citations = ResponseValidateNode().execute(state)["validated_citations"]

    assert len(citations) == 1, f"one endpoint cited three times reported as {citations}"
    assert len(citations) == len(set(citations))


# ── what the user actually reads ──────────────────────────────────────────────


def test_no_docs_message_is_bilingual() -> None:
    """A query that retrieves nothing is also the worst case for language detection.

    Guessing wrong there hands the reader a message in a language they may not read,
    so both languages are stated unconditionally rather than selected.
    """
    from src.nodes.response_generate_node import _NO_DOCS_ANSWER

    assert "No matching API documentation" in _NO_DOCS_ANSWER
    assert "見つかりませんでした" in _NO_DOCS_ANSWER


def test_offline_answer_says_it_is_not_an_answer() -> None:
    """The degraded branch must not read like a generated answer.

    It is reached on the platform whenever the LLM secrets are absent, and the run
    still reports success — so the text itself is the only signal the user gets.
    """
    from src.nodes.response_generate_node import _offline_answer

    text = _offline_answer([{"endpoint": "/v1/auth", "method": "POST"}])
    assert "could not be produced right now" in text
    assert "回答を生成できませんでした" in text
    assert "/v1/auth" in text, "the matching documentation should still be listed"


def test_offline_answer_with_no_citable_chunk_falls_back_to_the_no_docs_message() -> None:
    from src.nodes.response_generate_node import _NO_DOCS_ANSWER, _offline_answer

    assert _offline_answer([{"text": "something with no endpoint"}]) == _NO_DOCS_ANSWER


def test_offline_answer_still_passes_the_grounding_check() -> None:
    """Its citations must resolve, or the validator replaces it with a refusal."""
    state = {**_state([_CHUNK]), "raw_answer": _offline_for_chunk()}
    out = ResponseValidateNode().execute(state)
    assert out["grounding_confidence"] == 1.0
    assert out["final_answer"] == state["raw_answer"]


def _offline_for_chunk() -> str:
    from src.nodes.response_generate_node import _offline_answer

    return _offline_answer([_CHUNK])


# ── an unreachable index is not an empty one ──────────────────────────────────


def test_an_unreachable_kb_is_reported_as_such_not_as_no_documentation() -> None:
    """Two different facts, and only one of them is about the user's question.

    TargetedRetrieveNode fails CLOSED to an empty result so the pipeline keeps running,
    and by the time that empty result reaches generation it is indistinguishable from
    "nothing matched" — `kb_unavailable` is the only thing that tells them apart.

    Measured before this: an unreachable vector store produced "No matching API
    documentation found for this query." A reader concludes their API is undocumented
    when the truth is that nothing was looked up, and the run reports success.
    """
    from src.nodes.response_generate_node import _KB_UNAVAILABLE_ANSWER, _NO_DOCS_ANSWER

    out = ResponseGenerateNode().execute({"sanitized_query": "q", "retrieved_chunks": "[]", "kb_unavailable": True})
    assert out["raw_answer"] == _KB_UNAVAILABLE_ANSWER
    assert out["raw_answer"] != _NO_DOCS_ANSWER
    assert "could not be reached" in out["raw_answer"]
    assert "接続できなかった" in out["raw_answer"]
    assert "not a statement about whether your API is documented" in out["raw_answer"]


def test_the_unreachable_notice_never_calls_the_model() -> None:
    """Nothing was retrieved, so there is nothing to ground an answer on."""

    class _Boom:
        def generate(self, prompt):  # pragma: no cover - must not be reached
            raise AssertionError("the model was called with no index available")

    out = ResponseGenerateNode(llm_client=_Boom()).execute(
        {"sanitized_query": "q", "retrieved_chunks": "[]", "kb_unavailable": True}
    )
    assert out["status"] == "success"


def test_both_deterministic_notices_keep_full_confidence() -> None:
    """Neither asserts anything about an API, so neither can be ungrounded."""
    from src.nodes.response_generate_node import _KB_UNAVAILABLE_ANSWER, _NO_DOCS_ANSWER

    for notice in (_NO_DOCS_ANSWER, _KB_UNAVAILABLE_ANSWER):
        out = ResponseValidateNode().execute({"raw_answer": notice, "retrieved_chunks": []})
        assert out["grounding_confidence"] == 1.0
        assert out["final_answer"] == notice


# ── user-facing text must not leak internal architecture ──────────────────────


def test_no_user_facing_message_names_the_model_or_its_configuration() -> None:
    """The reader can act on none of it, and it says more about us than about them.

    The degraded notices used to read "no language model is configured" — one of them
    even named the AZURE_OPENAI_* secrets. The technical reason belongs in telemetry
    (the trace event, `llm_used`), where an operator looks for it; the person waiting
    for an answer gets told the answer could not be produced and what to do next.

    Scans the strings that reach a caller, not comments or trace payloads.
    """
    import ast
    import pathlib
    import re

    leaks = re.compile(
        r"language model|LLM\b|言語モデル|AZURE_OPENAI|ANTHROPIC_API_KEY|api[_ ]?key|secret",
        re.IGNORECASE,
    )
    offenders = []
    for path in pathlib.Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                name = getattr(target, "id", "")
                # Module-level notices this pipeline shows to a caller.
                if not re.search(r"ANSWER|NOTICE|GUIDANCE|MESSAGE|REJECTED", name):
                    continue
                text = " ".join(
                    n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                )
                if leaks.search(text):
                    offenders.append(f"{path.name}:{name}")
    assert not offenders, f"user-facing text names internal architecture: {offenders}"


def test_the_no_docs_sentence_follows_the_reader() -> None:
    """The sentence a reader gets when nothing matched, in their language.

    It used to be a fixed bilingual string, argued for on the grounds that a query short
    enough to retrieve nothing is where detection is least reliable. That holds for a reader
    whose language was never decided — still served both — but this agent DOES decide, per
    request, and ignoring a decision that was made is not caution.
    """
    from src.nodes.response_generate_node import _no_docs_answer

    assert _no_docs_answer("ja") == _NO_DOCS_JA
    assert _no_docs_answer("en") == _NO_DOCS_EN  # the control
    assert _no_docs_answer("") == _NO_DOCS_ANSWER  # no decision: say it in both


def test_the_node_gives_a_japanese_reader_the_japanese_sentence() -> None:
    """The wiring: the node reads the DECISION, not its own "en" prompt default."""
    state = {**_state([]), "query_language": "ja"}
    out = ResponseGenerateNode(llm_client=_Hallucinating()).execute(state)
    assert out["raw_answer"] == _NO_DOCS_JA


def test_a_state_with_no_decision_gets_both_languages() -> None:
    """The difference between "the reader wrote English" and "nobody decided".

    The node keeps an "en" default for the PROMPT — a model needs a language to write in —
    and that default must not leak into the notice: a mutant that used it there passed every
    test whose fixture set a language, which was all of them.
    """
    state = {k: v for k, v in _state([]).items() if k != "query_language"}
    out = ResponseGenerateNode(llm_client=_Hallucinating()).execute(state)
    assert out["raw_answer"] == _NO_DOCS_ANSWER
