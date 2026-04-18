"""spaCy-based text preprocessing pipeline."""

import re
import unicodedata
from typing import List

import spacy
from spacy.language import Language

from documind.utils import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Tokenizes and normalizes raw documents using spaCy.

    Applies lowercasing, punctuation removal, stop-word filtering,
    and lemmatization.  The processed token list feeds directly into
    FeatureExtractor.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

        try:
            self.nlp: Language = spacy.load(spacy_model, disable=["parser", "ner"])
        except OSError:
            logger.warning(
                "spaCy model '%s' not found — run: python -m spacy download %s",
                spacy_model,
                spacy_model,
            )
            raise

        self.nlp.max_length = 1_000_000
        logger.info("TextPreprocessor initialised with model '%s'", spacy_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """Return a cleaned, space-joined token string ready for vectorisation."""
        cleaned = self._clean_raw(text)
        tokens = self._tokenize(cleaned)
        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str], batch_size: int = 64) -> List[str]:
        """Process a list of documents in batches for efficiency."""
        cleaned = [self._clean_raw(t) for t in texts]
        results = []
        for doc in self.nlp.pipe(cleaned, batch_size=batch_size):
            results.append(" ".join(self._extract_tokens(doc)))
        return results

    def extract_entities(self, text: str) -> List[dict]:
        """Return named entities detected by spaCy (requires ner component)."""
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def get_pos_distribution(self, text: str) -> dict:
        """Return part-of-speech tag frequency distribution."""
        doc = self.nlp(text)
        distribution: dict = {}
        for token in doc:
            pos = token.pos_
            distribution[pos] = distribution.get(pos, 0) + 1
        total = sum(distribution.values()) or 1
        return {pos: count / total for pos, count in distribution.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_raw(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
        text = re.sub(r"\S+@\S+", " ", text)                  # emails
        text = re.sub(r"[^\w\s]", " ", text)                  # punctuation
        text = re.sub(r"\d+", " NUM ", text)                  # numbers → token
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def _tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return self._extract_tokens(doc)

    def _extract_tokens(self, doc) -> List[str]:
        tokens = []
        for token in doc:
            if self.remove_stopwords and token.is_stop:
                continue
            if token.is_punct or token.is_space:
                continue
            surface = token.lemma_ if self.lemmatize else token.text
            if len(surface) >= self.min_token_length:
                tokens.append(surface)
        return tokens
