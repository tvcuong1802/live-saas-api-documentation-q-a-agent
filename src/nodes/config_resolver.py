"""Resolve the agent manifest config regardless of how the framework passes it.

Production (LangGraph) passes a RunnableConfig: {"configurable": {...}}.
Unit tests pass the manifest directly: {"agent": {"config": {...}}}.
This helper reads both so manifest values are honored at runtime.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

_MANIFEST_KEYS = (
    "llm_model",
    "max_retry",
    "timeout_seconds",
    "retrieval",
    "configured_apis",
    "version_filter",
    "include_code_examples",
    "kb_backend",
    "grounding_threshold",
)


def resolve_agent_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return the agent-config dict from configurable, a passed manifest, or the file.

    An explicitly supplied agent config wins outright — it is not merged with the
    file. `{"agent": {"config": {}}}` therefore means "no settings", not "load the
    defaults from disk", which is what a caller passing an empty block intends and
    what the fail-closed tests rely on. The file is read only when the caller supplied
    no agent config at all, which is the production case: the Marketplace runner
    builds the agent with `agent_cls()` and invokes it with no agent config.
    """
    supplied = _from_caller(config)
    return _file_config() if supplied is None else supplied


def _from_caller(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """The agent config the caller supplied, or None if they supplied none.

    None and {} are deliberately different answers.
    """
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        ac = configurable.get("agent_config")
        if isinstance(ac, dict) and ac:
            return ac
        if any(k in configurable for k in _MANIFEST_KEYS):
            return configurable
    agent = config.get("agent")
    if isinstance(agent, dict):
        ac = agent.get("config")
        if isinstance(ac, dict):
            return ac
    if any(k in config for k in _MANIFEST_KEYS):
        return config
    # A bare RunnableConfig (thread_id and friends) carries no agent settings — that is
    # the absent case, not an empty one.
    return None


@lru_cache(maxsize=1)
def _file_config() -> dict[str, Any]:
    """Read config/config.yaml from the installed tree, or {} if it is not there.

    The Marketplace runner constructs the agent as ``agent_cls()`` with no arguments
    (verified in shared/bootstrap/marketplace_app.py) and its invoke path carries no
    agent config, so every value in config/config.yaml was unreachable in production
    even though the Dockerfile copies the file into the image. Reading it here — the
    single point every node already funnels through — is what makes the declared
    knobs take effect. Resolved from this file's location, not the process cwd, since
    the container runs from /app while tests run from anywhere.
    """
    path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    try:
        import yaml  # noqa: PLC0415 — optional at import time; absence just means no file config

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — a missing or malformed file must not break an invocation
        return {}
    agent = loaded.get("agent")
    if isinstance(agent, dict) and isinstance(agent.get("config"), dict):
        return dict(agent["config"])
    return dict(loaded) if isinstance(loaded, dict) else {}
