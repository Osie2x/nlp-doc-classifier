"""
DocumentClassifier — wraps the trained scikit-learn ensemble and exposes
a clean predict / predict_proba interface with full metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from documind.pipeline import FeatureExtractor, SentimentAnalyzer, TextPreprocessor
from documind.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    document_id: Optional[str]
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    sentiment: dict
    processing_time_ms: float
    tokens_extracted: int
    metadata: dict = field(default_factory=dict)


class DocumentClassifier:
    """
    End-to-end document classifier:
      raw text → preprocessing → feature extraction → ensemble prediction.

    Serialised artefacts (classifier, vectorizer, label encoder) are loaded
    from disk; the class is designed to be long-lived (load once, call many).
    """

    def __init__(
        self,
        model_path: str = "models/saved/documind_classifier.joblib",
        vectorizer_path: str = "models/saved/documind_vectorizer.joblib",
        label_encoder_path: str = "models/saved/documind_labels.joblib",
        spacy_model: str = "en_core_web_sm",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._label_encoder = None

        self.preprocessor = TextPreprocessor(spacy_model=spacy_model)
        self.feature_extractor = FeatureExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()

        if Path(model_path).exists():
            self.load(model_path, vectorizer_path, label_encoder_path)
        else:
            logger.warning("No saved model found at '%s' — train first.", model_path)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str, document_id: Optional[str] = None) -> ClassificationResult:
        start = time.perf_counter()

        processed = self.preprocessor.preprocess(text)
        features = self.feature_extractor.transform([processed])
        sentiment = self.sentiment_analyzer.analyze(text)

        proba = self._model.predict_proba(features)[0]
        class_idx = int(np.argmax(proba))
        classes = self._label_encoder.classes_
        predicted = classes[class_idx]
        confidence = float(proba[class_idx])

        prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ClassificationResult(
            document_id=document_id,
            predicted_category=predicted,
            confidence=round(confidence, 4),
            probabilities=prob_map,
            sentiment={
                "label": sentiment.label,
                "score": sentiment.score,
                "confidence": sentiment.confidence,
            },
            processing_time_ms=round(elapsed_ms, 2),
            tokens_extracted=len(processed.split()),
        )

    def predict_batch(
        self, texts: List[str], document_ids: Optional[List[str]] = None
    ) -> List[ClassificationResult]:
        if document_ids is None:
            document_ids = [None] * len(texts)

        start = time.perf_counter()
        processed = self.preprocessor.preprocess_batch(texts)
        features = self.feature_extractor.transform(processed)
        probas = self._model.predict_proba(features)
        classes = self._label_encoder.classes_

        results = []
        for i, (text, doc_id, proba) in enumerate(zip(texts, document_ids, probas)):
            class_idx = int(np.argmax(proba))
            sentiment = self.sentiment_analyzer.analyze(text)
            results.append(
                ClassificationResult(
                    document_id=doc_id,
                    predicted_category=classes[class_idx],
                    confidence=round(float(proba[class_idx]), 4),
                    probabilities={c: round(float(p), 4) for c, p in zip(classes, proba)},
                    sentiment={
                        "label": sentiment.label,
                        "score": sentiment.score,
                        "confidence": sentiment.confidence,
                    },
                    processing_time_ms=round(
                        (time.perf_counter() - start) / len(texts) * 1000, 2
                    ),
                    tokens_extracted=len(processed[i].split()),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str,
        vectorizer_path: str,
        label_encoder_path: str,
    ) -> None:
        self._model = joblib.load(model_path)
        self.feature_extractor.vectorizer = joblib.load(vectorizer_path)
        self.feature_extractor._is_fitted = True
        self._label_encoder = joblib.load(label_encoder_path)
        logger.info("Model artefacts loaded from '%s'", model_path)

    def is_ready(self) -> bool:
        return self._model is not None and self._label_encoder is not None

    @property
    def categories(self) -> List[str]:
        if self._label_encoder is None:
            return []
        return list(self._label_encoder.classes_)
