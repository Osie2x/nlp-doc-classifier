"""Unit tests for the data generator and model trainer."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGenerator:
    def test_generate_dataset_length(self):
        from scripts.generate_sample_data import generate_dataset
        records = generate_dataset(n=100, seed=0)
        assert len(records) == 100

    def test_generate_dataset_has_required_fields(self):
        from scripts.generate_sample_data import generate_dataset
        records = generate_dataset(n=10, seed=1)
        for r in records:
            assert "document_id" in r
            assert "category" in r
            assert "text" in r

    def test_generate_dataset_non_empty_text(self):
        from scripts.generate_sample_data import generate_dataset
        records = generate_dataset(n=20, seed=2)
        assert all(len(r["text"]) > 50 for r in records)

    def test_generate_dataset_categories(self):
        from scripts.generate_sample_data import generate_dataset, TEMPLATES
        records = generate_dataset(n=200, seed=3)
        observed = {r["category"] for r in records}
        assert observed == set(TEMPLATES.keys())

    def test_generate_dataset_reproducible(self):
        from scripts.generate_sample_data import generate_dataset
        r1 = generate_dataset(n=50, seed=42)
        r2 = generate_dataset(n=50, seed=42)
        assert [r["text"] for r in r1] == [r["text"] for r in r2]

    def test_unique_document_ids(self):
        from scripts.generate_sample_data import generate_dataset
        records = generate_dataset(n=100, seed=10)
        ids = [r["document_id"] for r in records]
        assert len(ids) == len(set(ids))


class TestModelTrainerInit:
    def test_trainer_init(self):
        from documind.models import ModelTrainer
        trainer = ModelTrainer(spacy_model="en_core_web_sm", max_features=100)
        assert trainer is not None
        assert trainer.metrics == {}

    def test_trainer_has_preprocessor(self):
        from documind.models import ModelTrainer
        trainer = ModelTrainer()
        assert trainer.preprocessor is not None

    def test_trainer_has_feature_extractor(self):
        from documind.models import ModelTrainer
        trainer = ModelTrainer()
        assert trainer.feature_extractor is not None


class TestDocumentClassifierInit:
    def test_classifier_not_ready_without_model(self, tmp_path):
        from documind.models import DocumentClassifier
        clf = DocumentClassifier(
            model_path=str(tmp_path / "nonexistent.joblib"),
            vectorizer_path=str(tmp_path / "v.joblib"),
            label_encoder_path=str(tmp_path / "l.joblib"),
        )
        assert not clf.is_ready()

    def test_classifier_has_components(self, tmp_path):
        from documind.models import DocumentClassifier
        clf = DocumentClassifier(
            model_path=str(tmp_path / "nonexistent.joblib"),
            vectorizer_path=str(tmp_path / "v.joblib"),
            label_encoder_path=str(tmp_path / "l.joblib"),
        )
        assert clf.preprocessor is not None
        assert clf.feature_extractor is not None
        assert clf.sentiment_analyzer is not None

    def test_categories_empty_before_load(self, tmp_path):
        from documind.models import DocumentClassifier
        clf = DocumentClassifier(
            model_path=str(tmp_path / "nonexistent.joblib"),
            vectorizer_path=str(tmp_path / "v.joblib"),
            label_encoder_path=str(tmp_path / "l.joblib"),
        )
        assert clf.categories == []
