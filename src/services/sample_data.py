"""Bundled sample data, so an agent can be TRIED before anyone has wired real data to it.

Byte-identical fleet-wide, for the same reason `kb_port.py` is: when CoE settles the real
data-loading mechanism this becomes one patch and a file copy, not fifty refactors.

The problem it solves, measured on 2026-09-07 with a real model bound: 48 of 69 agents
could not answer their own realistic question. Not because they were broken -- they were
fail-closed and correct -- but because nothing had supplied them with anything to reason
over. A person opening one on the Marketplace got "no source found", which demonstrates the
agent's honesty and nothing about its usefulness.

🔴 The thing this must NOT do is turn "I have no grounding" into a confident answer on
invented data. That trade would give away the property the fleet spent the most effort on.
So every use of a sample is announced: `using_sample_data` goes into state, the envelope
prints a notice above the answer, and `_provenance` names the data as unverified. The agent
becomes demonstrable without anyone being able to mistake the demonstration for a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SAMPLE_NOTICE_EN = (
    "⚠️ **Demonstration on bundled sample data.** Part of what this answer rests on was "
    "not supplied with the request, so an unverified sample shipped with the template "
    "stood in for it. Read this as a demonstration of what the agent produces, not as a "
    "verified result about your data."
)
_SAMPLE_NOTICE_JA = (
    "⚠️ **同梱のサンプルデータによるデモです。** 本回答の根拠の一部がリクエストに含まれ"
    "ていなかったため、テンプレート同梱の未検証サンプルで代用しています。エージェントの"
    "出力例であり、お客様のデータに関する確定した結果ではありません。"
)

#: Both halves. This is the form for the case that earns it -- the agent never settled on a
#: language, and this is the one line a reader must not miss, so it is shown in both rather
#: than in a guess.
SAMPLE_DATA_NOTICE = _SAMPLE_NOTICE_EN + "\n" + _SAMPLE_NOTICE_JA


def sample_data_notice(language: str) -> str:
    """The notice in the language the agent DECIDED on; both halves when it decided none.

    Measured on the pod 2026-09-16 (another template): an English question came back with an
    English answer under a two-paragraph notice, half of it Japanese, on a run where
    `answer_language` was `en` the whole time. Bilingual is the right answer to *not
    knowing*, and printing it once the answer is known is a different thing -- it hands the
    reader a paragraph they cannot read above the one they can.
    """
    code = str(language or "").strip().lower()[:2]
    if code == "ja":
        return _SAMPLE_NOTICE_JA
    if code == "en":
        return _SAMPLE_NOTICE_EN
    return SAMPLE_DATA_NOTICE


def notice_already_present(body: str) -> bool:
    """True when `body` already opens with the notice, in EITHER language.

    The de-dup check used to test only the first line of the bilingual constant, so once
    the notice became language-aware a Japanese reply would have been given a second one.
    """
    return any(variant.splitlines()[0] in body for variant in (_SAMPLE_NOTICE_EN, _SAMPLE_NOTICE_JA))


def sample_dir(repo_root: Path) -> Path:
    """Where samples live. Resolved from the caller's file, never from the cwd -- the Pod
    runs in /app and the tests run somewhere else."""
    return Path(repo_root) / "data" / "sample"


def load_sample(repo_root: Path, name: str) -> dict[str, Any] | None:
    """The sample packet named `name`, or None when the template ships none.

    Returns None rather than raising: a template with no sample is not broken, it simply
    cannot be demonstrated without data, and the caller's own "nothing was supplied"
    message is the right answer there.
    """
    path = sample_dir(repo_root) / f"{name}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def provenance(repo_root: Path) -> dict[str, Any]:
    """What the sample is and is not. Ships beside the data so the two cannot drift apart."""
    path = sample_dir(repo_root) / "_provenance.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}
