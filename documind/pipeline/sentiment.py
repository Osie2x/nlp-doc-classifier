"""Lightweight rule-based & lexicon-backed sentiment analyzer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from documind.utils import get_logger

logger = get_logger(__name__)

# Minimal built-in lexicons — production deployments can swap for VADER / TextBlob.
_POSITIVE_WORDS = frozenset(
    [
        "excellent", "outstanding", "superior", "approved", "compliant", "valid",
        "successful", "positive", "recommended", "favorable", "effective", "efficient",
        "innovative", "robust", "reliable", "accurate", "confirmed", "resolved",
        "improved", "beneficial", "advantageous", "satisfactory", "optimal",
    ]
)

_NEGATIVE_WORDS = frozenset(
    [
        "failure", "failed", "rejected", "non-compliant", "invalid", "error",
        "deficient", "inadequate", "violation", "breach", "risk", "adverse",
        "problematic", "defective", "unresolved", "concern", "issue", "critical",
        "negative", "poor", "insufficient", "denied", "penalty",
    ]
)

_INTENSIFIERS = frozenset(["very", "highly", "extremely", "severely", "critically"])
_NEGATORS = frozenset(["not", "no", "never", "neither", "nor", "without"])


@dataclass
class SentimentResult:
    score: float          # –1.0 (very negative) … +1.0 (very positive)
    label: str            # "positive" | "neutral" | "negative"
    confidence: float     # 0.0 … 1.0
    positive_count: int
    negative_count: int
    word_count: int


class SentimentAnalyzer:
    """
    Lexicon-based sentiment analyzer tailored for business documents.

    Accounts for intensifiers (+50 % weight) and negation windows
    (a negator flips the polarity of the next two sentiment words).
    """

    def __init__(
        self,
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
    ) -> None:
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        logger.info("SentimentAnalyzer initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> SentimentResult:
        tokens = self._tokenize(text)
        pos, neg = self._score_tokens(tokens)
        total = len(tokens) or 1
        raw_score = (pos - neg) / total
        score = max(-1.0, min(1.0, raw_score * 5))  # scale to [-1, 1]
        label = self._label(score)
        confidence = min(1.0, abs(score) + 0.3)
        return SentimentResult(
            score=round(score, 4),
            label=label,
            confidence=round(confidence, 4),
            positive_count=pos,
            negative_count=neg,
            word_count=total,
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        return [self.analyze(t) for t in texts]

    def sentiment_score(self, text: str) -> float:
        """Convenience wrapper returning only the float score."""
        return self.analyze(text).score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z]+\b", text.lower())

    def _score_tokens(self, tokens: List[str]) -> tuple[int, int]:
        pos = neg = 0
        negated = 0
        intensified = False

        for tok in tokens:
            if tok in _NEGATORS:
                negated = 2
                intensified = False
                continue

            boost = 1.5 if intensified else 1.0
            intensified = tok in _INTENSIFIERS

            if tok in _POSITIVE_WORDS:
                if negated:
                    neg += boost
                    negated -= 1
                else:
                    pos += boost
            elif tok in _NEGATIVE_WORDS:
                if negated:
                    pos += boost
                    negated -= 1
                else:
                    neg += boost
            elif negated:
                negated -= 1

        return int(pos), int(neg)

    def _label(self, score: float) -> str:
        if score >= self.positive_threshold:
            return "positive"
        if score <= self.negative_threshold:
            return "negative"
        return "neutral"
