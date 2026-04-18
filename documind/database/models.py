"""SQLAlchemy ORM models for DocuMind."""

from datetime import datetime, timezone

from .db import db


class Document(db.Model):
    """Raw document store."""

    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.String(128), unique=True, nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    word_count = db.Column(db.Integer, nullable=True)

    classifications = db.relationship(
        "ClassificationRecord", back_populates="document", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "source": self.source,
            "word_count": self.word_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ClassificationRecord(db.Model):
    """Stores every classification inference with full probability distribution."""

    __tablename__ = "classification_records"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(
        db.String(128),
        db.ForeignKey("documents.document_id"),
        nullable=False,
        index=True,
    )
    predicted_category = db.Column(db.String(64), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False)
    sentiment_label = db.Column(db.String(16), nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    probabilities_json = db.Column(db.Text, nullable=True)  # JSON string
    processing_time_ms = db.Column(db.Float, nullable=True)
    tokens_extracted = db.Column(db.Integer, nullable=True)
    model_version = db.Column(db.String(32), default="1.0.0")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    document = db.relationship("Document", back_populates="classifications")

    def to_dict(self) -> dict:
        import json
        return {
            "id": self.id,
            "document_id": self.document_id,
            "predicted_category": self.predicted_category,
            "confidence": self.confidence,
            "sentiment": {
                "label": self.sentiment_label,
                "score": self.sentiment_score,
            },
            "probabilities": json.loads(self.probabilities_json or "{}"),
            "processing_time_ms": self.processing_time_ms,
            "tokens_extracted": self.tokens_extracted,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelRun(db.Model):
    """Audit trail for training runs."""

    __tablename__ = "model_runs"

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(64), unique=True, nullable=False)
    model_version = db.Column(db.String(32), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    f1_macro = db.Column(db.Float, nullable=True)
    cv_mean = db.Column(db.Float, nullable=True)
    n_documents = db.Column(db.Integer, nullable=True)
    n_categories = db.Column(db.Integer, nullable=True)
    training_time_s = db.Column(db.Float, nullable=True)
    metrics_json = db.Column(db.Text, nullable=True)  # full report as JSON
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        import json
        return {
            "id": self.id,
            "run_id": self.run_id,
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "cv_mean": self.cv_mean,
            "n_documents": self.n_documents,
            "n_categories": self.n_categories,
            "training_time_s": self.training_time_s,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
