"""The mock backend must answer questions as users write them.

Measured before these tests existed: "How do I retrieve multiple records from a
kintone app?" scored 0.500 against the chunk that answers it exactly and was refused
at the 0.7 threshold, while the keyword form "kintone records" scored above it. The
existing unit test used the keyword form, so the suite reported the backend as
"usable at a real threshold" — usable only by someone who already knew the wording.

Two independent causes, pinned separately below: function words diluting the coverage
ratio, and a cosine-calibrated threshold applied to a lexical-coverage score.
"""

import pytest

from src.services.kb_service import KBService

_ON_TOPIC = [
    "How do I retrieve multiple records from a kintone app?",
    "How can I add a single record to a kintone app?",
    "Show me the freee deals endpoint",
    "list deals from freee",
]
_OFF_TOPIC = [
    "What is the weather today?",
    "Tell me a joke",
    "How do I reset my laptop password?",
]


@pytest.fixture
def kb() -> KBService:
    return KBService(config={"kb_backend": "mock"})


@pytest.mark.parametrize("query", _ON_TOPIC)
def test_a_natural_question_retrieves(kb: KBService, query: str) -> None:
    assert kb.search(query, [], top_k=5), query


@pytest.mark.parametrize("query", _OFF_TOPIC)
def test_an_off_topic_question_retrieves_nothing(kb: KBService, query: str) -> None:
    assert kb.search(query, [], top_k=5) == [], query


def test_function_words_do_not_dilute_the_score(kb: KBService) -> None:
    """The same content words must score the same however the question is padded."""
    bare = kb.search("retrieve multiple records kintone app", [], top_k=1, score_threshold=0.0)
    padded = kb.search(
        "So how do I retrieve multiple records from a kintone app please?", [], top_k=1, score_threshold=0.0
    )
    assert bare and padded
    assert bare[0]["score"] == padded[0]["score"]


def test_a_query_of_only_function_words_matches_nothing(kb: KBService) -> None:
    """Stripping stopwords must not leave an empty set that matches everything."""
    assert kb.search("how do I use it", [], top_k=5) == []


def test_the_mock_threshold_is_configurable_and_applied() -> None:
    """The floor is a declared value, not a constant the caller cannot reach.

    Bracketed with a query that lands mid-band: "kintone deals" names two APIs, so
    half of the corpus words it uses are missing from whichever document matched.
    """

    def _hits(threshold: float) -> int:
        kb = KBService(config={"kb_backend": "mock", "retrieval": {"mock_score_threshold": threshold}})
        return len(kb.search("kintone deals", [], top_k=5))

    assert _hits(0.4) > 0, "a permissive floor must admit the mid-band query"
    assert _hits(0.6) == 0, "a strict floor must exclude it — otherwise nothing is applied"


def test_on_topic_and_off_topic_bands_stay_separated(kb: KBService) -> None:
    """The threshold sits in a gap, not on top of the on-topic band.

    This is the assertion that fails if either the scorer or the corpus drifts far
    enough that the chosen floor stops separating the two populations.
    """

    def best(q: str) -> float:
        hits = kb.search(q, [], top_k=1, score_threshold=0.0)
        return hits[0]["score"] if hits else 0.0

    worst_on = min(best(q) for q in _ON_TOPIC)
    best_off = max(best(q) for q in _OFF_TOPIC)
    assert best_off < worst_on, f"bands overlap: off={best_off}, on={worst_on}"
