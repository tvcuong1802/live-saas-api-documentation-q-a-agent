# CMN-C1-071 — Live SaaS API Documentation Q&A Agent

> **Category**: Cat 1 (delivers a single technical capability; generic and use-case-agnostic)
> **Industry**: CMN

## Request format

Send the request as the message body. This is the same request the STG sign-off evidence sends and the live-path test exercises, so what is documented here is what is verified.

```json
{
  "input": "How do I retrieve multiple records from a kintone app?",
  "session_id": "stg-signoff-001"
}
```

## Overview

Answers questions about a SaaS provider's REST API from that provider's own documentation. Every factual claim in an answer must cite an endpoint that appears in a retrieved document; answers whose citations do not resolve are replaced by an explicit refusal rather than shown. When retrieval returns nothing the model is not called at all — there is nothing to ground an answer on, and asking anyway is how a RAG agent invents an API that does not exist.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent raises
`PlatformRequired` during graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Configuration

Runtime settings live in `config/config.yaml`; `config/agent.yaml` is the static
manifest. Nodes read the file when the caller supplies no agent config, which is the
case on the platform — the Marketplace runner constructs the agent with no arguments.

| Setting | Default | Effect |
|---------|---------|--------|
| `grounding_threshold` | `0.7` | Fraction of cited endpoints that must resolve to a retrieved chunk. Below it, the answer is replaced by a refusal. |
| `retrieval.top_k` | `5` | Maximum chunks returned per query. |
| `retrieval.score_threshold` | `0.7` | Similarity floor for the vector-store backend. |
| `retrieval.mock_score_threshold` | `0.5` | Lexical-coverage floor for the local/mock backend, which scores on a different scale. |
| `retrieval.stale_after_days` | `7` | Older chunks are answered with a staleness warning. |
| `kb_backend` | `mock` | `mock` uses the in-repo corpus; remove the key to use the real vector store. |
| `configured_apis` | `[]` | Restrict answers to these APIs. Empty means no restriction. |
| `include_code_examples` | `true` | Include code samples in the answer context. |

**Interim:** `kb_backend` is pinned to `mock` while the knowledge-base input mechanism
is decided. With no vector store provisioned the real path fails closed to zero chunks
and every question returns "no matching documentation". The in-repo corpus is
English-only, so a Japanese question matches only through Latin tokens it contains.

Secrets (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`)
are declared in `config/agent.yaml` and read through `ctx.secrets`. Without them the
agent still runs and says plainly that it could not generate an answer, listing the
documentation that matched.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/02_design.md` for the design and `docs/03_test_spec.md` for the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
