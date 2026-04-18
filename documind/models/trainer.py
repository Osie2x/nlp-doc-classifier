"""
ModelTrainer — full scikit-learn training pipeline with cross-validation,
hyperparameter search, and serialisation of all artefacts.

Ensemble: Logistic Regression + Linear SVM + Random Forest via soft voting.
Target accuracy: ≥ 82 % on 2,500+ record corpus.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from documind.pipeline import FeatureExtractor, TextPreprocessor
from documind.utils import ensure_dir, get_logger, timer

logger = get_logger(__name__)


class ModelTrainer:
    """
    Orchestrates the full training workflow:
      1. Preprocessing (spaCy)
      2. Feature extraction (TF-IDF + structural)
      3. Ensemble training (LR + SVM + RF)
      4. Cross-validated evaluation
      5. Artefact serialisation
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        max_features: int = 20_000,
        ngram_range: tuple = (1, 2),
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        model_dir: str = "models/saved",
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.model_dir = Path(model_dir)

        self.preprocessor = TextPreprocessor(spacy_model=spacy_model)
        self.feature_extractor = FeatureExtractor(
            max_features=max_features, ngram_range=ngram_range
        )
        self.label_encoder = LabelEncoder()
        self._model: Optional[VotingClassifier] = None
        self._metrics: dict = {}

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    @timer
    def train(
        self,
        texts: List[str],
        labels: List[str],
        evaluate: bool = True,
    ) -> Dict:
        logger.info("Starting training on %d documents", len(texts))

        # 1. Preprocess
        logger.info("Preprocessing documents …")
        processed = self.preprocessor.preprocess_batch(texts)

        # 2. Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # 3. Train/test split
        X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
            processed,
            encoded_labels,
            processed,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=encoded_labels,
        )

        # 4. Feature extraction
        logger.info("Extracting features …")
        X_train_feat = self.feature_extractor.fit_transform(X_train)
        X_test_feat = self.feature_extractor.transform(X_test)

        # 5. Build ensemble
        logger.info("Training ensemble classifier …")
        self._model = self._build_ensemble()
        self._model.fit(X_train_feat, y_train)

        # 6. Evaluate
        if evaluate:
            self._metrics = self._evaluate(X_test_feat, y_test, X_train_feat, y_train)
            self._log_metrics()

        # 7. Save
        self._save_artefacts()

        return self._metrics

    # ------------------------------------------------------------------
    # Ensemble construction
    # ------------------------------------------------------------------

    def _build_ensemble(self) -> VotingClassifier:
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=self.random_state,
            n_jobs=-1,
        )

        svm_base = LinearSVC(
            C=0.5,
            max_iter=2000,
            random_state=self.random_state,
        )
        svm = CalibratedClassifierCV(svm_base, cv=3)

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            random_state=self.random_state,
            n_jobs=-1,
        )

        return VotingClassifier(
            estimators=[("lr", lr), ("svm", svm), ("rf", rf)],
            voting="soft",
            weights=[2, 2, 1],
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, X_test, y_test, X_train, y_train) -> Dict:
        y_pred = self._model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            self._model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1
        )

        report = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
        )

        return {
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in cv_scores],
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "categories": list(self.label_encoder.classes_),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

    def _log_metrics(self) -> None:
        m = self._metrics
        logger.info("=" * 60)
        logger.info("  TRAINING COMPLETE")
        logger.info("  Test Accuracy  : %.2f%%", m["accuracy"] * 100)
        logger.info("  F1 (macro)     : %.4f", m["f1_macro"])
        logger.info("  CV Mean ± Std  : %.4f ± %.4f", m["cv_mean"], m["cv_std"])
        logger.info("  Train / Test   : %d / %d", m["n_train"], m["n_test"])
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save_artefacts(self) -> None:
        ensure_dir(self.model_dir)
        joblib.dump(self._model, self.model_dir / "documind_classifier.joblib")
        joblib.dump(self.feature_extractor.vectorizer, self.model_dir / "documind_vectorizer.joblib")
        joblib.dump(self.label_encoder, self.model_dir / "documind_labels.joblib")
        logger.info("Artefacts saved to '%s'", self.model_dir)

    @property
    def metrics(self) -> Dict:
        return self._metrics
