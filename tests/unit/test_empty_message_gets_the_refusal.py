"""An empty message is answered with what to send — not with a domain sentence.

`validation_error` is set by #pre_process and the shared envelope surfaces it, but only when
nothing else wrote a prose payload. The domain path used to run anyway on an empty query and
produce its own "nothing found" line, which the envelope quite correctly prefers — so the
reader was told about a search that should never have happened, and never learned what to
send. Measured through the runner's own call shape on five repos.

Written as one test across the layers: the per-layer suites each passed while the value died
between them, and a guard deleted from any single layer left every one of them green.
"""

from src.graph.graph import Graph
from src.nodes.main_node import MainNode
from src.nodes.response_validate_node import ResponseValidateNode

_REFUSAL = "There is no question here yet"


def _state(**over):
    base = {"user_input": "   ", "validation_error": "There is no question here yet. Tell me more."}
    base.update(over)
    return base


def test_the_domain_path_does_not_run_when_the_input_was_the_problem():
    out = MainNode().execute(_state())
    assert set(out) <= {"status"}, f"the domain path produced {sorted(set(out) - {'status'})}"


def test_post_process_does_not_compose_over_the_refusal():
    out = ResponseValidateNode().execute(_state())
    assert set(out) <= {"status"}, f"post_process produced {sorted(set(out) - {'status'})}"


def test_the_reader_is_told_what_to_send():
    envelope = Graph().get_output(_state())
    body = str(envelope.get("output") or "")
    assert _REFUSAL in body
    # NOT an assertion about `status`: the pipeline sets SUCCESS, and calling `get_output`
    # on a bare state would have me assert a value this call never produces. What matters
    # here is that the refusal is the body, and that nothing turned it into an error.
    assert str(envelope.get("status")) != "error"


def test_a_real_request_is_not_short_circuited():
    # The control. A guard that fired unconditionally would satisfy all three assertions
    # above while answering nothing at all.
    out = MainNode().execute({"user_input": "What does the accounts endpoint return?", "validation_error": ""})
    assert set(out) - {"status"}, "the domain path must still run for a real request"
