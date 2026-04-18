"""Unit tests for the Flask REST API."""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_classifier():
    clf = MagicMock()
    clf.is_ready.return_value = True
    clf.categories = ["Legal", "Medical", "Financial", "Technical", "HR", "Marketing", "Research", "Compliance"]

    from documind.models.classifier import ClassificationResult
    clf.predict.return_value = ClassificationResult(
        document_id="test-doc-001",
        predicted_category="Legal",
        confidence=0.87,
        probabilities={"Legal": 0.87, "Medical": 0.05, "Financial": 0.04, "Technical": 0.04},
        sentiment={"label": "neutral", "score": 0.01, "confidence": 0.31},
        processing_time_ms=42.5,
        tokens_extracted=18,
    )

    batch_result = ClassificationResult(
        document_id="doc-batch-1",
        predicted_category="Financial",
        confidence=0.91,
        probabilities={"Legal": 0.03, "Financial": 0.91, "Medical": 0.02, "Technical": 0.04},
        sentiment={"label": "positive", "score": 0.3, "confidence": 0.6},
        processing_time_ms=12.0,
        tokens_extracted=12,
    )
    clf.predict_batch.return_value = [batch_result]
    return clf


@pytest.fixture
def app(mock_classifier):
    with patch("documind.api.app.DocumentClassifier", return_value=mock_classifier):
        from documind.api import create_app
        application = create_app("config/config.yaml")
        application.config["TESTING"] = True
        application.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        with application.app_context():
            from documind.database import db
            db.create_all()
            yield application


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_json(self, client):
        data = client.get("/api/v1/health").get_json()
        assert data["status"] == "healthy"
        assert "model_ready" in data
        assert "version" in data

    def test_health_categories_present(self, client):
        data = client.get("/api/v1/health").get_json()
        assert isinstance(data["categories"], list)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

class TestCategoriesEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/categories")
        assert resp.status_code == 200

    def test_has_categories_key(self, client):
        data = client.get("/api/v1/categories").get_json()
        assert "categories" in data
        assert "total" in data

    def test_total_matches_list(self, client):
        data = client.get("/api/v1/categories").get_json()
        assert data["total"] == len(data["categories"])


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

class TestModelInfoEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200

    def test_has_model_fields(self, client):
        data = client.get("/api/v1/model/info").get_json()
        assert "model_type" in data
        assert "reported_accuracy" in data


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

class TestClassifyEndpoint:
    def test_classify_valid_text(self, client):
        resp = client.post(
            "/api/v1/classify",
            json={"text": "The defendant is hereby ordered to pay damages per the contract terms."},
        )
        assert resp.status_code == 200

    def test_classify_response_structure(self, client):
        data = client.post(
            "/api/v1/classify",
            json={"text": "The defendant is ordered to pay damages under the agreement."},
        ).get_json()
        assert "predicted_category" in data
        assert "confidence" in data
        assert "sentiment" in data
        assert "probabilities" in data

    def test_classify_missing_text(self, client):
        resp = client.post("/api/v1/classify", json={})
        assert resp.status_code == 400

    def test_classify_short_text(self, client):
        resp = client.post("/api/v1/classify", json={"text": "short"})
        assert resp.status_code == 400

    def test_classify_custom_document_id(self, client):
        data = client.post(
            "/api/v1/classify",
            json={"text": "Quarterly revenue exceeded projections by a significant margin.", "document_id": "custom-id-123"},
        ).get_json()
        assert data["document_id"] == "custom-id-123"

    def test_classify_non_string_text(self, client):
        resp = client.post("/api/v1/classify", json={"text": 12345})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Batch classify
# ---------------------------------------------------------------------------

class TestBatchClassifyEndpoint:
    def test_batch_valid(self, client):
        resp = client.post(
            "/api/v1/classify/batch",
            json={"documents": [{"text": "Quarterly earnings exceeded forecasts this period significantly."}]},
        )
        assert resp.status_code == 200

    def test_batch_response_structure(self, client):
        data = client.post(
            "/api/v1/classify/batch",
            json={"documents": [{"text": "Annual financial audit was completed with no findings."}]},
        ).get_json()
        assert "results" in data
        assert "total_processed" in data

    def test_batch_empty_list(self, client):
        resp = client.post("/api/v1/classify/batch", json={"documents": []})
        assert resp.status_code == 400

    def test_batch_missing_documents(self, client):
        resp = client.post("/api/v1/classify/batch", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200

    def test_stats_structure(self, client):
        data = client.get("/api/v1/stats").get_json()
        assert "total_documents" in data
        assert "total_classifications" in data
        assert "category_distribution" in data
