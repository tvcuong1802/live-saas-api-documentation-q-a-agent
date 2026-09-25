# CMN-C1-071 — Template Design Specification

**Template**: CMN-C1-071 — Live SaaS API Documentation Q&A Agent
**Category / Industry**: Cat 1 / CMN
**base_type / Level 2 type (catalog label only)**: `VectorRAGAgent` — this is the Level 2 base-agent label recorded in `config/agent.yaml`; the agent does **not** inherit this Level 2 base in code. Per the framework contract it inherits L1 (`AgentBaseGraph`) directly (see Design Decision Record).
**Scaffold issue**: agent1000/agent-templates/(internal issue reference removed) (PM Go 2026-05-22)

## Position in AgentCore Architecture

- **Agent Class**: `SaaSAPIDocQAAgent` (in `src/graph/graph.py`)
- **L1 Base**: **AgentBaseGraph** — L1-direct inheritance per the framework contract and the 2026-05-18 PM layer policy. The agent does **not** inherit from `VectorRAGAgent` / another template; `base_type: VectorRAGAgent` in `config/agent.yaml` is a catalog label only (see Design Decision Record). VectorRAG behaviour is implemented in the nodes directly, mirroring another template's config shape without importing it (§9 prohibits importing a sibling template).
- **Three-Layer Separation**:
  - **State**: flat `TypedDict` (`total=False`) extending `AgentState` — primitives + JSON-serializable types only (no Pydantic — msgpack incompatible).
  - **Node**: L1 inheritance (Template Method — override `execute(self, state, config=None) -> dict` only).
  - **Graph**: composition — `register_nodes()` + `add_edges()`.

## Architecture Overview

A Cat 1 VectorRAG agent that answers natural-language questions about a SaaS product's API documentation, retrieving and citing the relevant doc sections (with code examples and version awareness). Six logical nodes form the pipeline; security gates fire at fixed points (S-1/S-2 at the front, S-3 before the response exits).

### Node Configuration

| Node | Responsibility | Input State | Output State | Inherits |
|------|---------------|-------------|--------------|----------|
| `QueryNormalizeNode` | Detect language, normalize the query; **S-1 trust gate + S-2 input gate (injection/PII)** fire here, before any retrieval (fail-closed — raise on violation; no status flag) | `user_input` | `normalized_query`, `query_language`, `sanitized_query` | FunctionNode |
| `APINameDetectNode` | Extract target API / endpoint / method entities from the query | `sanitized_query` | `api_entities` (list of `{api_name, endpoint?, method?}`) | FunctionNode |
| `TargetedRetrieveNode` | Hybrid (dense+sparse) vector retrieval **scoped to the detected API**; applies version metadata filter; raises a staleness warning if indexed docs are older than the refresh interval | `sanitized_query`, `api_entities` | `retrieved_chunks`, `staleness_warning` | FunctionNode |
| `CodeExampleFilterNode` | Filter / rank retrieved chunks; surface code-example chunks; drop or flag outdated-version chunks | `retrieved_chunks` | `code_examples`, `retrieved_chunks` (filtered) | FunctionNode |
| `ResponseGenerateNode` | LLM answer (`gpt-4o-mini`, temp 0.2) grounded in the retrieved chunks, with inline citations | `sanitized_query`, `retrieved_chunks`, `code_examples` | `raw_answer` | FunctionNode |
| `ResponseValidateNode` | Validate citations are grounded in retrieved chunks; enforce confidence threshold; **S-3 output gate** fires here before the response exits | `raw_answer`, `retrieved_chunks` | `validated_citations`, `grounding_confidence`, `final_answer`, `result` | FunctionNode |

> **S-1 trust gate**: each node declares `required_trust_level = VERIFIED_EXTERNAL`; the framework `BaseNode.__call__` enforces it against `state["caller_trust_level"]` (ADR-006) — no graph-level `__pre_invoke__`. **S-2 input gate** (injection + credential scan, fail-closed) is inline in `QueryNormalizeNode` (pre_process). **S-3 output gate** (credential-leak scan, fail-closed) is inline in `ResponseValidateNode` (post_process). **S-4 trace** is the module-level `emit_trace_event` (`src/utils/audit.py`: `node_start` / `node_complete` / `node_error`) called on every node return path — the real framework `FunctionNode` has no `_emit_trace_event` hook (that was a the stub harness-only stub). **S-5** (no static credentials / Level-0 import) is enforced by the CI lint gates.

### Grounding validation rule (ResponseValidateNode — strict)

The citation-grounding check must be **endpoint-reference based**, not text-based, to close known bypass loopholes:

1. Build the allowed set `endpoint_refs` = the **unique** canonical references derived from each retrieved chunk's *metadata* only: `api_name@version/endpoint METHOD` (e.g. `kintone@v1//records GET`).
2. Extract `cited_endpoints` from `raw_answer` with a **strict regex** matching `/path METHOD` or `[endpoint: /path METHOD]`, **case-insensitive** (closes Loophole B = formatting variance, Loophole C = case).
3. A citation is grounded **iff** its `(endpoint, method)` matches an entry in `endpoint_refs`. Matching against `chunk_text` (Loophole A) or `source_path` is **forbidden** — those are not authoritative and are trivially spoofable by the LLM.
4. `validated_citations: list[str]` holds the **projected** grounded refs in the `api_name@version/endpoint METHOD` form (this is the stable assertion target for TC-08 / PB-1; resolves the moderator's projection carry-forward).
5. `grounding_confidence` = grounded citations / total cited, counted over **distinct** `(endpoint, method)` pairs on both sides of the ratio. Counting occurrences on one side and distinct pairs on the other made a repeated correct citation lower the score — one reference to `/v1/auth` scored 1.0, two scored 0.5, three scored 0.33 — so the answer that referenced its source most carefully was the one refused.
6. **An answer that cites nothing is not automatically grounded.** `total == 0` describes two opposite situations: the honest "no matching documentation" reply, and a model answering from its own weights. Only the first earns confidence 1.0, identified by exact equality with the fixed no-docs text; anything else with no citations scores 0.0. Before this distinction existed, an empty corpus produced a confident, entirely invented OAuth flow that passed the citation check because it cited nothing.
7. If confidence is below the configured threshold the answer is replaced by a refusal **before** S-3 fires. S-3 nonetheless scans the **raw** answer as well as the replacement: scanning only the replacement means a model that leaked a credential is reported as a routine low-confidence refusal rather than a security event.

### No-grounding path (ResponseGenerateNode)

When retrieval returns zero chunks the model is **not called**. There is nothing to ground an answer on, so asking anyway produces the classic RAG failure — an answer from the model's own weights, citing nothing. The node returns the fixed no-docs text, sets `no_grounding: true` in state, and emits `response_skipped_no_grounding`. The flag is declared in the state schema so it survives the LangGraph merge and appears in the checkpoint: a run that answered nothing still reports `status: success`, and the flag is what distinguishes the two after the fact.

Both the no-docs text and the degraded-LLM text are **fixed bilingual strings** (EN + JA), not translations selected from the detected language. A query short enough to retrieve nothing is also the case where language detection is least reliable, and guessing wrong hands the reader something they cannot read.

### Data Flow

```
START → initialize → QueryNormalize (S-1,S-2) → APINameDetect → TargetedRetrieve
        → CodeExampleFilter → ResponseGenerate → ResponseValidate (S-3) → finalize → END
```

Graph backbone slots (`pre_process` / `main` / `post_process`, mirroring another template): `QueryNormalize` maps to `pre_process`; `APINameDetect → TargetedRetrieve → CodeExampleFilter → ResponseGenerate` compose the `main` stage; `ResponseValidate` maps to `post_process`. Exact `register_nodes()` / `add_edges()` wiring is delivered in MR 3 (issue #9), including the `src/graph/__init__.py` export of `SaaSAPIDocQAAgent` (registry load — guards the another template DR-090-1 failure mode).

### State Definition

| Field | Type | Purpose | Required |
|-------|------|---------|----------|
| `user_input` | `str` | raw query (inherited from AgentState) | yes |
| `status` | `Any` | AgentStatus (inherited) | yes |
| `session_id` | `str` | session id (inherited) | yes |
| `node_history` | `list` | per-node trace (inherited) | auto |
| `error_log` | `list` | error records (inherited) | auto |
| `normalized_query` | `str` | language-normalized query | yes |
| `query_language` | `str` | detected language code | yes |
| `sanitized_query` | `str` | query after S-2 scrub | yes |
| `api_entities` | `Optional[str]` (JSON) | detected `{api_name, endpoint?, method?}`, JSON-serialized — **SEC-C1071-002** | yes |
| `retrieved_chunks` | `Optional[str]` (JSON) | chunks w/ metadata `api_name, version, endpoint, method, last_updated`, JSON-serialized — **SEC-C1071-002** | yes |
| `staleness_warning` | `str` | warning if indexed docs older than refresh interval | optional |
| `code_examples` | `Optional[str]` (JSON) | filtered code-example chunks, JSON-serialized — **SEC-C1071-002** | optional |
| `raw_answer` | `str` | LLM draft answer | yes |
| `validated_citations` | `list[str]` | citations grounded in retrieved chunks | yes |
| `grounding_confidence` | `float` | citation-grounding confidence (threshold-gated) | yes |
| `final_answer` | `str` | validated, S-3-passed answer | yes |
| `result` | `str` | final result surfaced to caller | yes |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serializable types).
- **SEC-C1071-002**: `api_entities`, `retrieved_chunks`, and `code_examples` are persisted as JSON strings (`Optional[str]`, via `src/utils/serde.encode_list`) rather than `list[dict[str, Any]]` — `Any`-typed dicts are not guaranteed msgpack-safe and could corrupt the LangGraph checkpoint. Consuming nodes decode with `src/utils/serde.decode_list`.
- No JWT / API keys / credentials in State (checkpoint DB leakage).
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic / dataclass / arbitrary objects (msgpack incompatible).

### Knowledge Base model

- **Self-managed** (not a shared KB): each deployment indexes its own configured APIs into its own Chroma collection via `scripts/load_kb.py`, mirroring another template's `configured_apis → collection` mapping.
- **Chunk metadata (mandatory)**: every chunk carries `api_name`, `version`, `endpoint`, `method`, `last_updated`. Outdated-version chunks are **filtered or flagged**, never silently returned (`version_filter: latest_only`).
- **Changelog priority**: changelog entries are indexed as priority chunks (higher retrieval weight).
- **Staleness warning**: at runtime, if the indexed docs for a target API are older than the configured refresh interval (weekly recommended), `staleness_warning` is set and surfaced in the response.

#### Local/mock backend retrieval scoring

`kb_backend: mock` ranks by lexical coverage rather than vector similarity, because `mock_embed` is an LCG hash with no semantic signal — cosine over it is meaningless. The score is the fraction of the query's *corpus-known content words* found in the chunk:

```
score = |q ∩ doc| / |(q ∩ V) − stopwords|        V = the union of all tokens in the corpus
```

Both sets in the denominator are **closed**: `V` is derived from the documents themselves and the stopword set is a fixed grammatical class. An earlier form divided by all query tokens, which let function words dilute the ratio — "How do I retrieve multiple records from a kintone app?" scored 0.500 against the chunk that answers it exactly and was refused, while the keyword form "kintone records" scored above the threshold. Subtracting a hand-maintained stopword list fixed the phrasings on the list and no others ("So … please?" still scored lower than the plain question); scoring against the corpus vocabulary ends the list.

The local backend also applies its **own** floor, `retrieval.mock_score_threshold` (default 0.5), rather than `retrieval.score_threshold`. The latter is calibrated for cosine similarity, and applying one number to two different metrics is what refused on-topic questions. Measured on `MOCK_API_DOCS`: on-topic natural questions score 1.000, off-topic ones 0.000, and a query naming two different APIs lands at 0.500 — the floor admits that middle case and excludes nothing else. The setting retires with the mock backend.

#### Interim KB pinning

`kb_backend` is pinned to `mock` in `config/config.yaml` while the knowledge-base input mechanism is decided (raised with PM/CoE, 2026-08). With no vector store provisioned the real path fails closed to zero chunks, so every question returns "no matching documentation" — an agent that is up and useless. When the input mechanism lands, removing that one line restores the real vector store with no code change.

**Known limitation:** the in-repo corpus is English-only. A Japanese question retrieves only through Latin tokens it happens to contain (an API name, an endpoint); it cannot match Japanese prose that is not in the corpus. This is a property of the interim data, not of the retrieval code, and is not worked around by adding invented Japanese documentation.

### S-2 rejection — what the caller reads

The gate is unchanged: `QueryNormalizeNode` raises `SecurityViolationError`, nothing is retrieved and no model is called. Only the presentation changed.

A rejected input used to return `{"output": null, "status": "error"}` — a blank screen with no reason — and the Marketplace runner turns any error status into `agent failed: RuntimeError: Graph invocation did not succeed`, which is worse. `get_output()` now returns a fixed bilingual refusal with `status: success`: the agent handled the request by declining it, and the caller can read why.

The rejection is identified by re-running **this template's own** S-2 predicates on the input, not by matching the framework's error text — that text is a framework detail and would silently stop matching if reworded, taking the refusal message with it. Requiring the input to trip our own detector also means an unrelated failure cannot be relabelled: a rejected input never reaches retrieval, so no other error can be in play.

**A genuine error is never dressed up as success.** An ordinary question that fails downstream keeps `status: error` and a null payload. That direction is the one worth guarding — a rule that tidies errors into success messages makes a broken agent look like a working one — and it is pinned by `test_a_genuine_error_is_not_dressed_up_as_success` plus a mutation that turns the guard into `if False`.

The refusal never echoes the input: reflecting rejected text back into whatever renders the answer is the second half of the injection it just declined.

### Runtime configuration resolution

Nodes read runtime settings through `resolve_agent_config()`, which accepts the LangGraph `RunnableConfig` shape, a directly-passed manifest, or nothing. When the caller supplies **no** agent config it falls back to `config/config.yaml`, resolved relative to the module rather than the process working directory.

This fallback is what makes the file take effect at all. The Marketplace runner constructs the agent as `agent_cls()` with no arguments (`shared/bootstrap/marketplace_app.py`) and its invoke path carries no agent config, so every value in `config/config.yaml` was unreachable in production even though the Dockerfile copies the file into the image — the declaration existed and the effect did not.

An **explicitly supplied** agent config wins outright and is not merged with the file: `{"agent": {"config": {}}}` means "no settings", which is what a caller passing an empty block intends and what the fail-closed backend tests rely on.

### Marketplace deployment contract

| Element | Requirement | Why |
|---|---|---|
| `cli.py` | `from src.graph.graph import SaaSAPIDocQAAgent` + `run_agent_marketplace(...)` | The platform imports this module to start the Pod; the entry point is derived from `class:` in `config/agent.yaml`, never hardcoded. |
| `pyproject.toml` | `agenticstar-agentcore[marketplace]==1.0.1` | `[marketplace]` supplies `agenticstar_platform`, which `shared.bootstrap.marketplace_app` imports. `[openai]` is deliberately absent — the registry wheel pins `langchain-core==1.3.3` while its own `[openai]` extra requires `>=1.4.3`, leaving the resolver no solution; that extra is installed in the image from the vendored wheel. |
| `src/services/llm_provider.py` | Build the client from `ctx.secrets` inside `execute()` | The runner passes no `config={"llm": ...}`, so the injected client is always `None` there and every LLM node took its degradation branch while still reporting success. Kept as a local, never assigned to `self` — one node instance serves concurrent invocations. |
| `config/agent.yaml` | `requires.secrets` lists all three Azure keys | Undeclared secrets bind to nothing, so the client fails to build and the agent degrades silently. `requires.extras` is the **registry** namespace validated against `shared.utils.extras_map`; `marketplace` is a packaging extra and is rejected there. |
| `src/api/server.py` | Same `ChainedSecretProvider(EnvProvider, factory)` chain as the runner | With the factory alone the adapter reads only `env/**/.env.*` and cannot see process environment variables, so a local run takes the degradation path while the platform takes the real one — and both report success. |

### Reusable: Live-Index Refresh node interface (factory requirement)

Per the (internal issue reference removed) factory spec, the live-index refresh interface is documented here as **reusable for DRAFT-452 / DRAFT-451**:

```python
# Refresh contract (implemented by scripts/load_kb.py + an optional refresh node)
def refresh_index(
    api_name: str,
    doc_path: str,
    collection: str,
    *,
    embedding_model: str,
    version_filter: str = "latest_only",
) -> dict:  # -> {"indexed": int, "skipped_stale": int, "last_updated": str}
    ...
```

Inputs: target API, doc source path, target collection, embedding model, version-filter policy. Output: counts + the newest `last_updated` seen (drives the staleness warning). The same signature is intended to back the live-index refresh in DRAFT-452 / DRAFT-451.

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (session_id, trust_level, credential handle) — via `config["configurable"]`
- [x] S-4 trace — module-level `emit_trace_event` (`src/utils/audit.py`), per-node `node_start`/`node_complete`/`node_error` (the real framework `FunctionNode` has no `_emit_trace_event` hook)
- [x] ConnectionPolicy (retry/timeout) — for LLM + vector store calls
- [x] SecurityViolationError
- [x] S-2 input gate (mandatory) — inline in `QueryNormalizeNode.execute()` (pre_process): injection + credential scan, fail-closed
- [x] S-3 output gate (mandatory) — inline in `ResponseValidateNode.execute()` (post_process): credential-leak scan, fail-closed

### Composition Pattern
- **Pattern**: Standalone (Cat 1 — independently deployable single capability).
- **Composition target**: none.
- **Error propagation strategy**: propagate — `SecurityViolationError` (S-2/S-3) is raised, not caught (the framework `BaseNode.__call__` re-raises security violations); other node errors set `status=ERROR` + `error_log` and emit `node_error`.

## Import Isolation Confirmation
- [x] Template does not import the agenticstar-platform SDK (Level 0).
- [x] Import targets: `framework/` and `shared/` only (no `agents/base/`; no sibling template import).

## Proof-of-Boundary Test Specification (PB-1..6)

Specified here so MR 4 implements them with stable assertions (the agent uses an LLM + a vector store, **no DB** — PB-3 mocks those, not a DB).

| PB | Boundary | Assertion |
|----|----------|-----------|
| PB-1 | Trace emitter | Each node emits `node_start` then `node_complete` (and `node_error` on the error path). Spy on the module-level `emit_trace_event` (patched in each node module); assert the event names fire on every return path. |
| PB-2 | State serialization | Post-invoke State is msgpack-serializable — all fields `str \| int \| float \| bool \| list \| dict \| None`; no Pydantic/dataclass/arbitrary objects. |
| PB-3 | LLM / vector-store fallback | Mock `LLMClient.generate()` timeout and the vector store returning HTTP 429 → status `ERROR`, `error_log` populated, `node_error` emitted. (No DB mock — agent has no DB.) |
| PB-4 | Import isolation | AST scan of `src/`: 0 imports of Level 0 (`agenticstar` / `import agenticstar`). |
| PB-5 | Checkpoint safety | Saved checkpoint State contains no PII, credentials, `InvocationContext`, or a persisted `formatted_prompt`. |
| PB-6 | Lifecycle execution order | Spy-verify the node run order QueryNormalize → APINameDetect → TargetedRetrieve → CodeExampleFilter → ResponseGenerate → ResponseValidate. |

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed 6-node pipeline (not a think→act→observe loop). |
| Inheritance | L1-direct (`AgentBaseGraph`) | L2 (`VectorRAGAgent`/another template) | **L1-direct** | the framework contract + criterion #1/#10 (strict) + 2026-05-18 PM policy mandate L1-direct; §9 prohibits importing a sibling template. Architect note suggesting `VectorRAGAgent` inheritance predates the 2026-05-18 layer change (confirmed stale by CoE 2026-06-10). `base_type: VectorRAGAgent` retained as a catalog label only. |
| Vector store | Chroma (local) | pgvector / pinecone | **Chroma (default, switchable)** | Canonical for VectorRAG (another template); runs locally, no external service for dev/test. |
| KB ownership | Self-managed (`load_kb.py`) | Shared KB | **Self-managed** | Cat 1 generic — customer configures their own API stack. |
| Composition | Standalone | GraphNode | **Standalone** | Independently deployable Cat 1 capability. |
| No-citation answer | Treat as grounded | Treat as ungrounded unless it is the fixed no-docs text | **Fixed text only** | "Cites nothing" describes both an honest refusal and a hallucination; only the deterministic refusal is distinguishable, so only it earns the free pass. |
| Empty retrieval | Call the LLM anyway | Skip the call | **Skip** | Nothing to ground on, and the output must be discarded regardless — skipping is both safer and cheaper. |
| Mock retrieval floor | Reuse `score_threshold` | Separate `mock_score_threshold` | **Separate** | Lexical coverage and cosine similarity are different scales; one number cannot be calibrated for both. |
| Coverage denominator | All query tokens | Corpus-known content words | **Corpus-known** | A stopword list never finishes; the corpus vocabulary is a closed set derived from the data. |
| Config when caller passes none | Hardcoded defaults | Read `config/config.yaml` | **Read the file** | The Marketplace runner passes no config, so hardcoded defaults made every declared setting unreachable in production. |

## Language of the trailer

The notice is printed in the language of the answer, decided by
`language_of_answer()` in `src/services/disclaimer.py`. That module is the MECHANISM and is
byte-identical in every template; what varies per agent is the POLICY, declared as
`LANGUAGE_POLICY` in `src/services/agent_scope.py` beside the scope wording.

The mechanism does not ask "does a Latin letter appear". A translated Japanese answer
legitimately carries Latin names — a standard's abbreviation, an endpoint, the agent id —
and counting those as English made a fully Japanese document ask for a bilingual trailer.
So Latin counts only as prose: words of four letters or more that are neither an ALL-CAPS
acronym, nor quoted machine output, nor markup the agent declares. When both languages
carry substance the trailer is bilingual, which is the safe direction.

This agent overrides `ignore_patterns` with its citation-marker syntax (`[endpoint: ... ]`). Measured on the Japanese case in `deploy/disclaimer_cases.json`: 53 Japanese characters against 52 of Latin, all of it marker text — no threshold separates those two numbers, so the markers are declared as markup instead.

The cases this was measured on are in `deploy/disclaimer_cases.json`, one per language and
one per failure path; `tests/unit/test_language_of_answer.py` pins the mechanism itself.

### A marker inside a code fence is data, not a notice

`has_disclaimer()` searched the whole answer, so an agent whose reply ends with a JSON
block carrying a `disclaimer` field looked as if it already had a trailer and got none
appended — measured on another template, 2026-08-28. Markers are now looked for in the answer's
prose, with fenced blocks removed. Shared mechanism, so the fix is one file in every
template; `tests/unit/test_language_of_answer.py` pins it.

## Which language the answer is written in (2026-08-28)

`query_language` came from a character check: any kana or kanji meant Japanese, anything
else meant English. A question typed in romanised Japanese is entirely Latin, so the
reader who wrote `kintone apuri kara fukusuu no rekoodo wo shutoku suru ni wa` was
answered in English (measured 2026-08-28).

The shared intake call (`src/services/input_intake.py`) now decides, and the character
check is its fallback. The call sits **inside `execute()`, after the S-2 gate** in the
same node — nothing that failed the injection or credential scan is ever sent to the model
provider. `query_normalized` records the source, so a degraded run reads as degraded.

No fields are extracted: retrieval runs on the normalized query and nothing here consumes
a structured field. The endpoint citation markers this agent stamps on every claim are
declared in `LANGUAGE_POLICY` so a Japanese answer is not mistaken for a bilingual one.

### Shared-file round, 2026-09-17

Three fleet-shared modules were taken back to the canonical copy, which had moved ahead of
this repo by 215 lines. Two behaviours change for a reader:

- **A credential-shaped value never reaches the screen.** `with_disclaimer()` now derives
  the outgoing text once, after the answer exists and before any caller sees it, and
  withholds it when it carries an access-token shape (`sk-proj-`, `sk_live_`, a JWT,
  `AKIA…`, `ghp_…`, `glpat-…`, `xox…`). A credential that came from the SENDER is reported
  as such and delivered as `success`, because they can act on it; one the agent produced
  stays an `error` for an operator. The framework's S-3 gate covers the shapes it knows and
  does not know these (measured on wheel 1.0.3).
- **An empty structured payload is said in words, not shown as JSON.** When the renderer
  returns "" — every field of the report is empty — the envelope no longer falls back to
  the raw payload, which was putting the serialised object back on screen.

The bundled-sample notice now follows the language the agent decided on, and stays bilingual
only where no decision was made.
