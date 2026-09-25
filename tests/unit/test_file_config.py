"""config/config.yaml must actually reach the nodes.

The Marketplace runner builds the agent with `agent_cls()` and invokes it without an
agent config (shared/bootstrap/marketplace_app.py:95). Every tunable in
config/config.yaml was therefore unreachable in production while the Dockerfile
faithfully copied the file into the image — the declaration existed, the effect did
not, and nothing reported the gap. These tests fail if that silence comes back.
"""

from pathlib import Path

import pytest
import yaml

from src.nodes.config_resolver import _file_config, resolve_agent_config

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@pytest.fixture(autouse=True)
def _clear_cache():
    _file_config.cache_clear()
    yield
    _file_config.cache_clear()


def test_config_yaml_exists_and_parses() -> None:
    assert _CONFIG.is_file(), "config/config.yaml is missing — every runtime knob is dead"
    assert isinstance(yaml.safe_load(_CONFIG.read_text()), dict)


def test_declared_values_reach_a_caller_that_passes_nothing() -> None:
    """Production shape: no config at all."""
    declared = yaml.safe_load(_CONFIG.read_text())["agent"]["config"]
    resolved = resolve_agent_config(None)
    for key, value in declared.items():
        assert resolved[key] == value, key


def test_declared_values_reach_a_bare_runnable_config() -> None:
    """LangGraph shape: a RunnableConfig carrying no agent config."""
    declared = yaml.safe_load(_CONFIG.read_text())["agent"]["config"]
    resolved = resolve_agent_config({"configurable": {"thread_id": "t-1"}})
    assert resolved["grounding_threshold"] == declared["grounding_threshold"]


def test_an_explicit_caller_config_wins_outright(tmp_path: Path) -> None:
    """An explicit block replaces the file rather than layering over it.

    `{"agent": {"config": {}}}` means "no settings" — the fail-closed tests pass
    exactly that to reach the no-backend path, and would silently stop testing it if
    the file's kb_backend leaked in underneath.
    """
    resolved = resolve_agent_config({"agent": {"config": {"grounding_threshold": 0.42}}})
    assert resolved == {"grounding_threshold": 0.42}
    assert resolve_agent_config({"agent": {"config": {}}}) == {}


def test_editing_the_file_changes_the_resolved_value(monkeypatch) -> None:
    """The strongest form: the file is read, not mirrored by a hardcoded default.

    Both sit at 0.7 today, so an assertion on the number alone would pass even if the
    file were never opened.
    """
    original = _CONFIG.read_text()
    try:
        doc = yaml.safe_load(original)
        doc["agent"]["config"]["grounding_threshold"] = 0.31
        _CONFIG.write_text(yaml.safe_dump(doc))
        _file_config.cache_clear()
        assert resolve_agent_config(None)["grounding_threshold"] == 0.31
    finally:
        _CONFIG.write_text(original)
        _file_config.cache_clear()
