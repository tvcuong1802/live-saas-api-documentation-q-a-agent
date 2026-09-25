# CMN-C1-071 — Test Specification

Template: CMN-C1-071 — Live SaaS API Documentation Q&A Agent
Run: `PYTHONPATH=the stub harness pytest tests/ -v`

## 1. Framework Compliance Tests (TC-01..08)

Canonical location: `tests/unit/test_compliance.py` (S-1/S-2/S-3 promoted from MR !3).

| TC-ID | Test | Location | Expected |
|-------|------|----------|----------|
| TC-01 | State is a flat TypedDict; populated state is msgpack-safe | `test_compliance.py::TestTC01StateContract` | All values are primitives / list / dict / None |
| TC-02 | `SecurityViolationError` on prompt injection (S-2) | `TestTC02TC06InputGate::test_tc02_injection_raises` | Raises |
| TC-03 | No JWT / credentials in State values | `TestTC03NoCredsInState` | `detect_credentials` over state values = none |
| TC-04 | `InvocationContext` read from `config["configurable"]` only | `TestTC04ContextFromConfigOnly` | Stray `invocation_context` in state ignored; no config → fail-closed |
| TC-05 | `_emit_trace_event` recorded on nodes | `TestTC05TraceRecorded` | `node_start` + `node_complete` captured |
| TC-06 | `_security_gate_input` non-bypassable | `TestTC02TC06InputGate::test_tc06_gate_input_runs_on_clean` | Clean input returns unchanged; gate always runs |
| TC-07 | `_security_gate_output` non-bypassable (S-3 in `invoke()`) | `TestTC07OutputGate` | Credential in `final_answer` → raises; clean passes, `result` projected |
| TC-08 | `required_trust_level` declared + enforced (S-1) | `TestTC08TrustLevel` | Missing ctx / ANONYMOUS → raises; VERIFIED_EXTERNAL passes |

## 2. Proof-of-Boundary Tests (PB-1..6)

| PB-ID | Boundary | Location | Expected |
|-------|----------|----------|----------|
| PB-1 | BaseNode → EventEmitter (`_emit_trace_event`) | `test_pb_cmn_c1_071.py::TestPB1TraceEmitter` | `node_start`/`node_complete` on success; `node_error` on failure |
| PB-2 | State serialization (msgpack-safe) | `test_pb_cmn_c1_071.py::TestPB2StateSerialization` | Post-pipeline state values are primitives only |
| PB-3 | LLM / vector-store fallback (no DB) | `test_pb_cmn_c1_071.py::TestPB3Fallback` | LLM raise + vector-store 429 → `status=ERROR` + `error_log` + `node_error`; `KBBackendUnavailableError` message reaches `error_log` |
| PB-4 | Import isolation (no Level 0) | `test_import_isolation.py` *(extended from scaffold baseline)* | AST scan of `src/`: 0 `agenticstar` imports |
| PB-5 | Checkpoint / State safety | `test_state_safety.py` *(extended from scaffold baseline)* | No credential-named fields, no Pydantic / InvocationContext in State |
| PB-6 | Lifecycle execution order | `test_pb_cmn_c1_071.py::TestPB6LifecycleOrder` | 6-node `node_start` order via manual slot-chain (ci_stub `_invoke_impl` is a no-op; MainNode emits none) |

## 3. Unit Tests (node / service)

| File | Coverage |
|------|----------|
| `tests/unit/test_nodes.py` | Happy + error path for each of the 6 nodes |
| `tests/unit/test_main_node.py` | MainNode orchestration (mock KB backend) + error propagation |
| `tests/unit/test_compliance.py` | TC-01..08 (above) |
| `tests/unit/test_grounding.py` | Citation grounding, the no-grounding path, and the bilingual user-facing messages |
| `tests/unit/test_retrieval_scoring.py` | Local/mock retrieval: natural-question recall, off-topic rejection, threshold application |
| `tests/unit/test_file_config.py` | `config/config.yaml` reaching the nodes, and explicit config taking precedence |
| `tests/unit/test_input_rejection_output.py` | S-2 rejection surfacing as a readable bilingual refusal, and a genuine error staying an error |

### 3.1 Grounding and retrieval — what each test pins

Every defect below was reachable while the suite was green, so each entry names the
observable behaviour rather than the function it calls.

| Test | Pins |
|------|------|
| `test_validator_grants_full_confidence_only_to_the_exact_no_docs_answer` | An answer produced with zero retrieved chunks does not receive confidence 1.0 unless it is the fixed no-docs text. Before this, an empty corpus shipped an invented OAuth flow at full confidence. |
| `test_empty_retrieval_never_reaches_the_model` | With nothing retrieved, the model is not called at all — the refusal is produced deterministically rather than requested and then discarded. |
| `test_repeating_a_correct_citation_does_not_lower_the_score` | Confidence is computed over distinct `(endpoint, method)` pairs. Previously 1 citation scored 1.0, 2 scored 0.5, 3 scored 0.33. |
| `test_validated_citations_list_has_no_duplicates` | The reported citation list is a set of sources, not a tally of mentions. Found by mutation-testing: removing the de-duplication left every scoring test green. |
| `test_no_docs_message_is_bilingual` / `test_offline_answer_says_it_is_not_an_answer` | The two texts a user reads when the agent cannot answer state both languages, and the degraded text does not read like a generated answer. |
| `test_a_natural_question_retrieves` | Questions phrased as users write them retrieve. The prior test used the keyword form `"kintone records"`, which passed while natural questions were refused. |
| `test_function_words_do_not_dilute_the_score` | Padding a question with function words does not change its score. |
| `test_on_topic_and_off_topic_bands_stay_separated` | The threshold sits in the gap between the two populations, not inside the on-topic band. |
| `test_a_genuine_error_is_not_dressed_up_as_success` | An ordinary question that failed downstream still reports failure. The refusal rule must not tidy real errors into success messages — mutating its guard to `if False` fails this test. |
| `test_the_refusal_does_not_echo_the_input` | The rejection message does not reflect the rejected text back to the caller. |
| `test_editing_the_file_changes_the_resolved_value` | `config/config.yaml` is read, not mirrored by a hardcoded default that happens to hold the same number. |

**Mutation testing.** Each fix above was verified by reverting it in place and
confirming the suite fails, then restoring it. Two mutants survived on the first
attempt and are recorded because they changed what the tests assert: removing the
citation de-duplication left all scoring tests green (the ratio's numerator and
denominator repeat together, so only the reported list changes), and the
`declared-secrets` release gate accepted a manifest it should have rejected.

**Note:** `src/api/server.py` is validated at STG deploy (requires the real
agenticstar-platform runtime; `compile()`/`provision_secrets()` are not in
the stub harness), so it is intentionally not exercised in `run-tests`.

## Refused input — what the sender receives (shared contract, 2026-09-15)

Measured across the fleet with a real model: a message the framework's S-2 gate declined
came back as `status: error` carrying the generic line "No answer could be produced for
this request." `normalize_terminal_output()` raises on any status but SUCCESS, so the
runner discarded the whole envelope and the sender read **"agent failed"** — with nothing
to act on, and no reason to send anything different next time.

| Situation | What is returned | Why |
|---|---|---|
| S-2 declined the MESSAGE | `status: success`, `refusal_kind: "input"`, a sentence naming what to change, plus the trailer | The sender is legitimate and holds something they can fix; they only learn that if the reply reaches them |
| The agent has its own refusal wording | That wording, not the shared sentence | "The shipment could not be classified" says which step stopped; the generic line does not |
| S-1 denied the CALLER | `status: error`, `refusal_kind: "trust"`, the refusal and nothing else | A caller not permitted to invoke the agent must not be told what it is for |
| S-3 blocked the agent's OWN output | unchanged — `status: error` | The agent produced something its output gate would not pass. The sender can do nothing with that, and must not be invited to retry |
| The agent genuinely broke | unchanged — `status: error` | The one signal that says this is an operations problem |

Nothing downstream reads `status` to detect a refusal any more: the envelope names the
refusal in `refusal_kind`. A contract that could only be read by the symptom it was fixing
was not a contract.

Enforced by `tests/unit/test_disclaimer_always_present.py` —
`test_a_refused_MESSAGE_is_delivered_and_says_what_to_change`,
`test_a_REAL_failure_is_still_an_error` (its control), and
`test_the_gate_token_is_matched_as_a_whole_token`.

## Credential scan — key shapes (added 2026-09-15)

| Input | Expected | Why |
|---|---|---|
| `sk-` + 20 alnum (classic) | flagged | the only shape the suite used to exercise |
| `sk-proj-...` | flagged | **OpenAI's current default**. Missed before this round: the pattern required N alnum characters straight after `sk-`, so the hyphen ended the match |
| `sk-svcacct-...` | flagged | service-account keys, same mechanism |
| `sk-live_...` | flagged | underscore in the body, same mechanism |
| `risk-assessment-checklist-template-v2` | NOT flagged | contains the literal `sk-`; a false positive costs a legitimate sender their answer |
| `sk-` followed by dashes/underscores only | NOT flagged | an already-redacted value — the sender did the right thing |

Enforced by `tests/unit/test_credential_scan_shapes.py`, which reads the patterns back out
of the production source instead of restating them, and fails if a listed file stops
declaring one. Counting patterns is not measuring: the previous set had four patterns and
missed three of the four shapes in use.

### Shared-file round, 2026-09-17

`tests/unit/test_disclaimer_always_present.py` and `tests/unit/test_input_intake.py` are
the canonical copies again; they carry the credential-egress cases and the intake cases
added fleet-wide since this repo last took them.
