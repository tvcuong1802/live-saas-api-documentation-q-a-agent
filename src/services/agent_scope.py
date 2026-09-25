"""What THIS agent's output must not be used for.

Kept in its own module so the trailer's wording lives beside the agent it
describes, while the MECHANISM stays byte-identical fleet-wide in
``disclaimer.py``. A generic "AI-generated draft" is true of every template
here and tells a reader nothing they can act on; this names the decision the
output must not stand in for.
"""

from __future__ import annotations

SCOPE_EN = "Reference only: an answer assembled from the API documentation that matched your question. It is not a specification, and it does not guarantee the endpoint behaves this way in your tenant or version. Confirm against the official documentation before you build on it."

SCOPE_JA = "参考情報です。ご質問に一致した API ドキュメントから構成した回答であり、仕様書ではなく、お客様のテナントやバージョンで同じ挙動になることを保証するものではありません。実装前に公式ドキュメントでご確認ください。"


# How this agent's answers are read for language, on top of the shared defaults in
# disclaimer.DEFAULT_LANGUAGE_POLICY. Empty means the defaults were measured to hold for
# THIS agent -- see deploy/disclaimer_cases.json, which carries the cases they were
# measured on. Populate a key here when this agent's output shape differs, e.g.:
#   "identifiers": ("Physical AI",)   # mixed-case names it keeps in Latin inside Japanese
#   "min_chars": 60                   # answers routinely embed a short foreign quotation
#   "dominance": 0                    # never call a mixed answer single-language
#   "ignore_quoted_code": False       # this agent's fenced blocks hold human language
LANGUAGE_POLICY: dict[str, object] = {
    # Every answer stamps citation markers -- "[endpoint: /k/v1/records GET]" -- outside code
    # fences, one per claim. Measured on the Japanese case in deploy/disclaimer_cases.json:
    # 53 Japanese characters against 52 of Latin, and every one of those Latin words came
    # from the markers ("kintone", "endpoint", "records"). No threshold separates those two
    # numbers; what separates them is that the markers are markup, not English.
    "ignore_patterns": (r"\[endpoint:[^\]]*\]",),
}


# What the intake call reads out of a free-form question. Mechanism is byte-identical
# fleet-wide in input_intake.py; the closed sets below are this agent's.
#
# No `fields` are declared: retrieval runs on the normalized query and nothing here
# consumes a structured field. What the call adds is the language of the answer --
# measured 2026-08-28, a question typed in romanised Japanese was answered in English,
# because every character in it is Latin.
INTAKE_POLICY: dict[str, object] = {
    "languages": ("en", "ja"),
    "default_language": "en",
    "fields": {},
    "capabilities": (
        "Answer a question about a connected SaaS API from its documentation",
        "Show a request example for an endpoint",
        "Point to the documentation behind a specific behaviour",
    ),
    "examples": (
        {
            "message": "kintone apuri kara fukusuu no rekoodo wo shutoku suru ni wa dou sureba yoi desu ka",
            "expect": {"language": "ja", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            "message": "kintone アプリから複数のレコードを取得するにはどうすればよいですか？",
            "expect": {"language": "ja", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            # An English question naming a Japanese product: the product is not the language.
            "message": "How do I retrieve multiple records from a kintone app?",
            "expect": {"language": "en", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            "message": "Deploy this change to our production tenant for us.",
            "expect": {"language": "en", "fields": {}, "fits": "no", "suggestion": 1},
        },
    ),
}
