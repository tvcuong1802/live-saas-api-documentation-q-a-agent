"""AgentCore Platform v1.0"""

# Service layer: domain queries, external API wrappers, data aggregation.
# Must NOT contain business logic, routing, or credentials.
# Nodes call this; this calls shared/services/ for external integrations.
#
# NOTE: scaffold-standard placeholder. CMN-C1-071 is a Cat 1 retrieval agent —
# its domain/service logic lives in src/services/kb_service.py. This generic
# Service stub is kept for scaffold-structure (criterion #14) and is intentionally
# unused; implement only if a non-KB domain service is later required.

from __future__ import annotations

from typing import Any


class Service:
    """Domain service."""

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch domain data for the given query."""
        # TODO: implement domain service logic
        raise NotImplementedError("Implement fetch() for Service")
