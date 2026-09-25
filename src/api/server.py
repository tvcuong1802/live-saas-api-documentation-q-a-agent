"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, AgentGateway calls agent.invoke() directly.
#
# NOTE: compile() / provision_secrets() are real agenticstar-platform runtime
# methods (not in the stub harness); this module is validated at STG deploy, not in
# run-tests (see docs/07_operation_guide.md). fastapi is a deploy dependency.

import os
import secrets
from uuid import uuid4

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.invocation_context import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory

# The registry wheel CI installs does not ship shared.secrets.chained_provider or
# shared.secrets.env_provider; the vendored wheel used for the image does. A
# module-level import therefore turns run-tests red on CI while passing locally —
# measured on another template, pipeline 77932: 13 failures, all ModuleNotFoundError.
#
# Degrade instead of breaking: without the chain this adapter reads only env/**/.env.*
# files, which is exactly the behaviour it had before the chain was added.
try:
    from shared.secrets.chained_provider import ChainedSecretProvider
    from shared.secrets.env_provider import EnvProvider
except ImportError:  # registry wheel — no chained provider available
    ChainedSecretProvider = None
    EnvProvider = None
from src.graph.graph import Graph

app = FastAPI(title="Agent")

_NAMESPACE = "agent1000/agent-templates"
_AGENT_NAME = "cmn_c1_071"

agent = Graph()
agent.compile()
# Same provider chain the platform runner builds (shared/bootstrap/marketplace_app.py).
# With secrets_factory alone this adapter reads only env/**/.env.* files and cannot see
# process environment variables, so a local run takes the degradation path while the
# platform takes the real one — and both report success, which is what makes the
# difference expensive to find. Mirroring the chain keeps the two paths comparable.
_base_provider = secrets_factory(namespace=_NAMESPACE, agent_name=_AGENT_NAME)
if ChainedSecretProvider is not None and EnvProvider is not None:
    agent.provision_secrets(
        ChainedSecretProvider(
            EnvProvider(namespace=_NAMESPACE, agent_name=_AGENT_NAME),
            _base_provider,
            namespace=_NAMESPACE,
            agent_name=_AGENT_NAME,
        )
    )
else:
    # Registry wheel: no chain available. Reads env/**/.env.* only — the
    # behaviour this adapter had before the chain was added.
    agent.provision_secrets(_base_provider)


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone/STG caller auth ((internal issue reference removed), the deployment runbook): when
    # INVOKE_AUTH_TOKEN is set on the server environment, callers that no upstream
    # middleware vouched for (still ANONYMOUS) must present it as a Bearer token
    # and run at VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary (standalone equivalent of the
    # platform AuthMiddleware) — a deployment-level caller credential, not an agent
    # secret, so ctx.secrets does not apply (no InvocationContext exists before
    # auth); see the framework contract "Entry-point exception".
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of a clean 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        # Framework contract: invoke(user_input, session_id=None, ctx=None). The S-1 gate and
        # nodes read InvocationContext from config["configurable"]["invocation_context"].
        # agent.invoke() is Any (the wheel ships no py.typed); the contract is a
        # result dict, so pin it here rather than leaking Any to the caller.
        result: dict[str, Any] = agent.invoke(req.input, ctx=ctx)
        return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "cmn_c1_071"}
