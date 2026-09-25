"""Knowledge Base retrieval + ingestion service for CMN-C1-071.

Self-managed vector KB over a SaaS product's API documentation.

Fail-closed in production: if chromadb is unavailable and `kb_backend` is not
"mock" and no `docs=` corpus is injected, KBService raises
KBBackendUnavailableError rather than silently serving empty/mock results.
The deterministic mock corpus + `mock_embed` are test-only / `kb_backend: mock`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

try:
    import chromadb  # type: ignore[import-not-found]

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    chromadb = None

logger = logging.getLogger(__name__)


class KBBackendUnavailableError(RuntimeError):
    """Raised when no usable KB backend is available in production."""


# Test-only / kb_backend:mock corpus. Chunks carry the mandatory metadata.
MOCK_API_DOCS: list[dict[str, Any]] = [
    {
        "api_name": "kintone",
        "version": "v1",
        "endpoint": "/k/v1/records",
        "method": "GET",
        "last_updated": "2026-05-01",
        "text": "GET /k/v1/records retrieves multiple records from an app. "
        "Example: curl -H 'X-Cybozu-API-Token: ***' '/k/v1/records?app=1'",
        "chunk_type": "code_example",
    },
    {
        "api_name": "kintone",
        "version": "v1",
        "endpoint": "/k/v1/record",
        "method": "POST",
        "last_updated": "2026-05-01",
        "text": "POST /k/v1/record adds a single record to an app. Body: {app, record}.",
        "chunk_type": "reference",
    },
    {
        "api_name": "freee",
        "version": "v1",
        "endpoint": "/api/1/deals",
        "method": "GET",
        "last_updated": "2026-04-15",
        "text": "GET /api/1/deals lists accounting deals for a company.",
        "chunk_type": "reference",
    },
]


def mock_embed(text: str) -> list[float]:
    """Deterministic 64-dim unit embedding (LCG). Test / mock-backend only."""
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) or 1
    a, c_val, m = 1103515245, 12345, 2**31
    cur = seed
    vec = []
    for _ in range(64):
        cur = (a * cur + c_val) % m
        vec.append((cur / m) * 2.0 - 1.0)
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


# Floor for the lexical-coverage score of the local/mock backend. Measured on
# MOCK_API_DOCS with the vocabulary-scoped scorer: on-topic natural questions score
# 1.000, off-topic ones 0.000, and a query naming two different APIs lands at 0.500.
# The floor admits that middle case and excludes nothing else.
_MOCK_SCORE_THRESHOLD = 0.5


class KBService:
    """Retrieval over the API-doc KB. Constructor-injectable for tests."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        chroma_client: Any = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        docs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or {}
        self._vocab: set[str] | None = None
        self.embed_fn = embed_fn or mock_embed
        self._chroma_client = chroma_client
        retrieval = self.config.get("retrieval", {}) if isinstance(self.config, dict) else {}
        self.kb_backend = self.config.get("kb_backend") or retrieval.get("kb_backend")
        self.collection = retrieval.get("vector_store", {}).get("collection", "saas_api_docs")

        if docs is not None:
            self.docs: list[dict[str, Any]] | None = docs
            self._mode = "local"
        elif self.kb_backend == "mock":
            self.docs = MOCK_API_DOCS
            self._mode = "local"
        elif chroma_client is not None or _CHROMA_AVAILABLE:
            self.docs = None
            self._mode = "chroma"
        else:
            raise KBBackendUnavailableError(
                "chromadb is not installed and kb_backend != 'mock'. Install chromadb "
                "for production, set kb_backend: mock for local dev, or inject docs= in tests."
            )

    def search(
        self,
        query: str,
        api_entities: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Return scored chunks (metadata + text + score), API-scoped + thresholded."""
        api_entities = api_entities or []
        if self._mode == "local":
            return self._local_search(query, api_entities, top_k, score_threshold)
        return self._chroma_search(query, api_entities, top_k, score_threshold)

    def _scope(self, chunks: list[dict[str, Any]], api_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        names = {str(e.get("api_name", "")).lower() for e in api_entities if e.get("api_name")}
        if not names:  # empty entities -> unscoped (graceful)
            return chunks
        return [c for c in chunks if str(c.get("api_name", "")).lower() in names]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t}

    # Function words carry no retrieval signal and never appear in an API reference.
    # They are excluded from the QUERY side of the coverage ratio: the score is the
    # fraction of query tokens found in the chunk, so every "how do I ... from a ..."
    # pushed a perfectly on-topic question below the threshold. Measured before the
    # change: "How do I retrieve multiple records from a kintone app?" scored 0.500
    # against the chunk that answers it exactly, and was refused at the 0.7 threshold,
    # while the keyword query "kintone records" scored above it. The unit test used
    # the keyword form, so the suite agreed the backend was "usable at a real
    # threshold" — it was usable only by someone who already knew the wording.
    _STOPWORDS = frozenset(
        """a an and are as at be by can do does for from get how i in is it me my of on or
        please show that the there these this to use using want way we what when where which
        who why will with you your""".split()
    )

    def _vocabulary(self) -> set[str]:
        """Every token appearing anywhere in the local corpus, computed once."""
        if self._vocab is None:
            self._vocab = set()
            for c in self.docs or []:
                self._vocab |= self._tokens(
                    " ".join(str(c.get(k, "")) for k in ("text", "api_name", "endpoint", "method"))
                )
        return self._vocab

    def _local_search(
        self, query: str, api_entities: list[dict[str, Any]], top_k: int, score_threshold: float
    ) -> list[dict[str, Any]]:
        # Lexical relevance for the local/mock backend. mock_embed is an LCG hash
        # with no semantic signal, so cosine over it is meaningless; score instead
        # by the fraction of query tokens present in the chunk (0..1).
        # The local backend scores lexical coverage; `score_threshold` is calibrated for
        # the vector store's cosine similarity. Applying one number to two different
        # metrics is what refused on-topic questions: measured on this corpus, on-topic
        # queries and off-topic ones separate completely (1.000 vs 0.000), but 0.7 sat
        # inside the on-topic band under the scorer in use at the time.
        # Overridable via retrieval.mock_score_threshold; it retires with the mock KB.
        retrieval = self.config.get("retrieval", {}) if isinstance(self.config, dict) else {}
        threshold = float(retrieval.get("mock_score_threshold", _MOCK_SCORE_THRESHOLD))
        scoped = self._scope(self.docs or [], api_entities)
        q_tokens = self._tokens(query)
        # Denominator = the query tokens this corpus could possibly match: its own
        # vocabulary, minus function words. Both sets are CLOSED — the vocabulary is
        # derived from the documents and the stopwords are a fixed grammatical class —
        # so no word has to be added by hand as new phrasings appear. An earlier form
        # subtracted stopwords only, and "So how do I ... please?" still scored lower
        # than the same question asked plainly, because "so" was not on the list. That
        # is a list that never finishes; scoring against the vocabulary ends it.
        q_content = (q_tokens & self._vocabulary()) - self._STOPWORDS
        scored = []
        for c in scoped:
            if not q_content:
                s = 0.0
            else:
                doc_tokens = self._tokens(
                    " ".join(str(c.get(k, "")) for k in ("text", "api_name", "endpoint", "method"))
                )
                s = len(q_content & doc_tokens) / len(q_content)
            if s >= threshold:
                scored.append({**c, "score": s})
        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored[:top_k]

    def _chroma_search(
        self, query: str, api_entities: list[dict[str, Any]], top_k: int, score_threshold: float
    ) -> list[dict[str, Any]]:
        client = self._chroma_client
        if client is None:
            if not _CHROMA_AVAILABLE:
                raise KBBackendUnavailableError("chromadb unavailable at query time")
            client = chromadb.Client()
        # get_COLLECTION, not get_or_create: creating one on the READ path is how a Pod
        # with chromadb installed but no corpus ingested comes back with an empty list
        # that is indistinguishable from "no matching documentation". The reader is then
        # told the knowledge base has nothing on their question, when the truth is that
        # the knowledge base was never there. Creating is the ingestion service's job.
        try:
            collection = client.get_collection(self.collection)
        except Exception as exc:  # noqa: BLE001 -- chroma raises its own type here
            raise KBBackendUnavailableError(
                f"collection {self.collection!r} does not exist on this backend -- the corpus has not been ingested"
            ) from exc
        if collection.count() == 0:
            raise KBBackendUnavailableError(
                f"collection {self.collection!r} is empty -- the corpus has not been ingested"
            )
        names = [str(e.get("api_name", "")).lower() for e in api_entities if e.get("api_name")]
        where = {"api_name": {"$in": names}} if names else None
        res = collection.query(query_embeddings=[self.embed_fn(query)], n_results=top_k, where=where)
        out: list[dict[str, Any]] = []
        metadatas = (res.get("metadatas") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        for meta, doc, dist in zip(metadatas, documents, distances):
            score = 1.0 - float(dist)
            if score >= score_threshold:
                out.append({**(meta or {}), "text": doc, "score": score})
        return out


class KBIngestionService(KBService):
    """Write path — kept separate from the read-only KBService at runtime."""

    def refresh_index(
        self,
        api_name: str,
        doc_path: str,
        collection: str,
        *,
        embedding_model: str,
        version_filter: str = "latest_only",
    ) -> dict[str, Any]:
        """Reusable live-index refresh contract (factory: DRAFT-452/451).

        Returns {"indexed": int, "skipped_stale": int, "last_updated": str}.
        """
        import json
        import os

        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"doc_path not found: {doc_path}")
        with open(doc_path, encoding="utf-8") as f:
            chunks = json.load(f)

        latest = ""
        for c in chunks:
            lu = str(c.get("last_updated", ""))
            if lu > latest:
                latest = lu
        kept = [c for c in chunks if str(c.get("api_name", "")).lower() == api_name.lower()]
        skipped = len(chunks) - len(kept) if version_filter == "latest_only" else 0

        if self._chroma_client is not None or _CHROMA_AVAILABLE:
            client = self._chroma_client or chromadb.Client()
            col = client.get_or_create_collection(collection)
            col.add(
                ids=[f"{api_name}-{i}" for i in range(len(kept))],
                embeddings=[self.embed_fn(str(c.get("text", ""))) for c in kept],
                documents=[str(c.get("text", "")) for c in kept],
                metadatas=[{k: v for k, v in c.items() if k != "text"} for c in kept],
            )
        return {"indexed": len(kept), "skipped_stale": skipped, "last_updated": latest}
