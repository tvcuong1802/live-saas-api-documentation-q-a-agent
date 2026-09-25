"""
src/utils/security.py — CMN-C1-071

In-repo, public-safe S-2 (input) and S-3 (output) detection helpers.

These replace the non-public ``framework.security`` module (the stub harness-only →
ImportError in production; CoE SEC-PRE-01). All detection is plain in-repo regex.
"""

from __future__ import annotations

import re

# Prompt-injection markers (S-2 input scan).
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+.*instructions", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"bypass\s+security", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+developer", re.IGNORECASE),
    re.compile(r"ignore\s+.*safety", re.IGNORECASE),
    re.compile(r"指示を無視"),
    re.compile(r"安全ガイドラインを無視"),
]

# Credential / token markers (S-2 input + S-3 output scan).
_CREDENTIAL_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{19,}"),
    re.compile(r"eyJ[a-zA-Z0-9._-]{10,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]{20,}"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*=\s*[\"'][^\"']{8,}[\"']"),
]


def detect_injection(text: str) -> bool:
    """True if the text contains a prompt-injection marker."""
    return any(p.search(text or "") for p in _INJECTION_PATTERNS)


def detect_credentials(text: str) -> bool:
    """True if the text contains any credential / token pattern."""
    return any(p.search(text or "") for p in _CREDENTIAL_PATTERNS)
