"""Financial sentiment scoring for the news feed.

The default ``LexiconSentiment`` is dependency-free and deterministic — a
finance-aware lexical baseline good enough to wire up the full sentiment pipeline
(ingest -> score -> store -> API -> UI) and tests without pulling in heavyweight
models. Finance text needs a *finance-specific* lexicon: general-purpose word
lists misread terms like "liability", "cost" or "tax" (neutral in finance, not
negative). This list is **Loughran–McDonald-inspired** (the standard interpretable
financial lexicon).

For genuinely contextual scoring, fine-tune/ship **FinBERT** behind the same
``SentimentAnalyzer`` interface (a Phase-3 deliverable) — it understands that
"strong sell" ≠ "strong". The interface and the ``NewsArticle.sentiment`` column
stay identical either way, exactly like ``HashingEmbedder`` →
``SentenceTransformerEmbedder`` for semantic search (see ``embeddings.py``).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

# Scores in [-1, 1]. Anything inside this neutral band classifies as "neutral".
NEUTRAL_BAND = 0.05

_TOKEN_RE = re.compile(r"[a-z]+")
# A negator flips the polarity of the next polarity word within this many tokens.
_NEGATION_WINDOW = 3

# Loughran–McDonald-inspired financial word lists (common inflections included so
# exact-token matching catches headline forms without a stemmer). Kept as
# whitespace-split strings so the word groups stay readable and black-stable.
POSITIVE: frozenset[str] = frozenset(
    (
        "gain gains gained rise rises rose rising rally rallies rallied "
        "rebound rebounds rebounded surge surges surged soar soars soared "
        "jump jumps jumped climb climbs climbed profit profits profitable "
        "beat beats beating exceed exceeds exceeded outperform outperforms outperformed "
        "top tops topped growth grow grows growing grew expansion expand expands expanding "
        "boom booming strong stronger strength robust record high highs upside "
        "upbeat optimism optimistic bullish upgrade upgrades upgraded boost boosts boosted "
        "recovery recover recovers recovered recovering approval approved approves "
        "win wins won success successful breakthrough positive surplus"
    ).split()
)

NEGATIVE: frozenset[str] = frozenset(
    (
        "loss losses lose loses losing lost fall falls fell falling "
        "drop drops dropped dropping decline declines declined declining "
        "plunge plunges plunged slump slumps slumped tumble tumbles tumbled "
        "crash crashes crashed sink sinks sank slide slides slid plummet plummets plummeted "
        "weak weaker weakness weakening downturn recession slowdown slowing "
        "contraction shrink shrinks shrank miss misses missed shortfall deficit "
        "default defaults defaulted bankrupt bankruptcy insolvent insolvency "
        "layoff layoffs recall recalls fail fails failed fear fears worry worries worried "
        "concern concerns concerned fraud scandal probe lawsuit sanction sanctions sanctioned "
        "downgrade downgrades downgraded warning warn warns warned crisis turmoil selloff panic "
        "bearish pessimism pessimistic gloom gloomy struggle struggles struggled struggling "
        "halt halts halted risk risks volatile volatility"
    ).split()
)

# Function-word negators (not sentiment-bearing themselves).
NEGATORS: frozenset[str] = frozenset("not no never without neither nor lack lacks lacking".split())

# Single source of truth for O(1) polarity lookup.
_POLARITY: dict[str, int] = {w: 1 for w in POSITIVE} | {w: -1 for w in NEGATIVE}


@runtime_checkable
class SentimentAnalyzer(Protocol):
    name: str

    def score(self, texts: list[str]) -> list[float]: ...


class LexiconSentiment:
    """Finance-aware lexical sentiment (dependency-free baseline).

    Per text: tokenize, count positive/negative polarity hits (a negator flips the
    next polarity word within ``_NEGATION_WINDOW`` tokens), and return
    ``(pos - neg) / (pos + neg)`` in [-1, 1]; ``0.0`` when no polarity words occur.
    Not contextual — swap in a FinBERT-backed analyzer for nuance.
    """

    name = "lexicon"

    def score(self, texts: list[str]) -> list[float]:
        return [self._score_one(t) for t in texts]

    @staticmethod
    def _score_one(text: str) -> float:
        pos = neg = 0
        negate_for = 0  # tokens remaining in an active negation window
        for token in _TOKEN_RE.findall((text or "").lower()):
            if token in NEGATORS:
                negate_for = _NEGATION_WINDOW
                continue
            polarity = _POLARITY.get(token, 0)
            if polarity:
                if negate_for > 0:
                    polarity = -polarity
                    negate_for = 0  # one flip per negator
                if polarity > 0:
                    pos += 1
                else:
                    neg += 1
            if negate_for > 0:
                negate_for -= 1
        total = pos + neg
        return 0.0 if total == 0 else (pos - neg) / total


def get_default_analyzer() -> SentimentAnalyzer:
    """The dependency-free default. Override at call sites to use a real model."""
    return LexiconSentiment()


def classify(score: float | None) -> str:
    """Bucket a polarity score: positive / neutral / negative (unknown if None)."""
    if score is None:
        return "unknown"
    if score > NEUTRAL_BAND:
        return "positive"
    if score < -NEUTRAL_BAND:
        return "negative"
    return "neutral"
