"""TF-IDF feature extraction with optional structural feature augmentation."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from documind.utils import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Builds a rich feature matrix combining TF-IDF bag-of-words with
    hand-crafted structural features (document length, average word length,
    punctuation density, numeral ratio, etc.).

    The combined representation gives the downstream classifier both
    lexical and surface-level signals.
    """

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
        use_structural_features: bool = True,
    ) -> None:
        self.use_structural_features = use_structural_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            analyzer="word",
        )
        self._is_fitted = False
        logger.info(
            "FeatureExtractor initialised (max_features=%d, ngrams=%s)",
            max_features,
            ngram_range,
        )

    # ------------------------------------------------------------------
    # Fit / transform API  (mirrors scikit-learn convention)
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "FeatureExtractor":
        self.vectorizer.fit(texts)
        self._is_fitted = True
        logger.info("Vectorizer fitted on %d documents", len(texts))
        return self

    def transform(self, texts: List[str]):
        self._check_fitted()
        tfidf = self.vectorizer.transform(texts)
        if self.use_structural_features:
            structural = self._structural_features(texts)
            return hstack([tfidf, csr_matrix(structural)])
        return tfidf

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return self.transform(texts)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        self._check_fitted()
        names = list(self.vectorizer.get_feature_names_out())
        if self.use_structural_features:
            names += self._structural_feature_names()
        return names

    def vocabulary_size(self) -> int:
        self._check_fitted()
        return len(self.vectorizer.vocabulary_)

    # ------------------------------------------------------------------
    # Structural / hand-crafted features
    # ------------------------------------------------------------------

    def _structural_features(self, texts: List[str]) -> np.ndarray:
        records = [self._extract_structural(t) for t in texts]
        return np.array(records, dtype=np.float32)

    def _extract_structural(self, text: str) -> List[float]:
        words = text.split()
        n_words = len(words) or 1
        n_chars = len(text) or 1
        avg_word_len = sum(len(w) for w in words) / n_words
        unique_word_ratio = len(set(words)) / n_words
        num_count = len(re.findall(r"\bNUM\b", text))
        num_ratio = num_count / n_words
        upper_ratio = sum(1 for c in text if c.isupper()) / n_chars
        return [
            n_words,
            n_chars,
            avg_word_len,
            unique_word_ratio,
            num_ratio,
            upper_ratio,
        ]

    @staticmethod
    def _structural_feature_names() -> List[str]:
        return [
            "feat_word_count",
            "feat_char_count",
            "feat_avg_word_len",
            "feat_unique_word_ratio",
            "feat_num_ratio",
            "feat_upper_ratio",
        ]

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before use.")
