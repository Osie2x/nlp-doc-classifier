"""
DocuMind REST API — v1 routes.

Endpoints
---------
GET  /api/v1/health                  — liveness + model status
GET  /api/v1/categories              — supported document categories
POST /api/v1/classify                — classify a single document
POST /api/v1/classify/batch          — classify up to 100 documents
GET  /api/v1/documents               — list stored documents (paginated)
GET  /api/v1/documents/<id>          — document + classification history
DELETE /api/v1/documents/<id>        — delete document record
GET  /api/v1/stats                   — aggregate platform statistics
GET  /api/v1/model/info              — model metadata
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from functools import wraps

from flask import Blueprint, current_app, jsonify, request

from documind.database import ClassificationRecord, Document, db
from documind.utils import get_logger

logger = get_logger(__name__)
api_bp = Blueprint("api", __name__)

MAX_BATCH = 100
MAX_TEXT_CHARS = 50_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classifier():
    return current_app.config["CLASSIFIER"]


def _error(message: str, status: int = 400) -> tuple:
    return jsonify({"error": message, "status": status}), status


def require_classifier(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _classifier().is_ready():
            return _error(
                "Model not loaded. Run 'python scripts/train_model.py' first.", 503
            )
        return f(*args, **kwargs)
    return wrapper


def _validate_text(text) -> str | None:
    if not text or not isinstance(text, str):
        return "Field 'text' is required and must be a string."
    if len(text.strip()) < 10:
        return "Field 'text' must contain at least 10 characters."
    if len(text) > MAX_TEXT_CHARS:
        return f"Field 'text' exceeds maximum length of {MAX_TEXT_CHARS} characters."
    return None


# ---------------------------------------------------------------------------
# Health & meta
# ---------------------------------------------------------------------------

@api_bp.route("/health", methods=["GET"])
def health():
    clf = _classifier()
    return jsonify({
        "status": "healthy",
        "model_ready": clf.is_ready(),
        "categories": clf.categories,
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@api_bp.route("/categories", methods=["GET"])
def categories():
    return jsonify({
        "categories": _classifier().categories,
        "total": len(_classifier().categories),
    })


@api_bp.route("/model/info", methods=["GET"])
@require_classifier
def model_info():
    clf = _classifier()
    return jsonify({
        "model_type": "ensemble (LogisticRegression + LinearSVC + RandomForest)",
        "categories": clf.categories,
        "n_categories": len(clf.categories),
        "nlp_pipeline": "spaCy en_core_web_sm",
        "feature_extraction": "TF-IDF (unigrams + bigrams) + structural features",
        "reported_accuracy": "82%",
        "version": "1.0.0",
    })


# ---------------------------------------------------------------------------
# Classification endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/classify", methods=["POST"])
@require_classifier
def classify():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    err = _validate_text(text)
    if err:
        return _error(err)

    doc_id = body.get("document_id") or str(uuid.uuid4())
    source = body.get("source")

    result = _classifier().predict(text, document_id=doc_id)

    _persist_record(doc_id, text, source, result)

    return jsonify({
        "document_id": doc_id,
        "predicted_category": result.predicted_category,
        "confidence": result.confidence,
        "sentiment": result.sentiment,
        "probabilities": result.probabilities,
        "tokens_extracted": result.tokens_extracted,
        "processing_time_ms": result.processing_time_ms,
    }), 200


@api_bp.route("/classify/batch", methods=["POST"])
@require_classifier
def classify_batch():
    body = request.get_json(silent=True) or {}
    documents = body.get("documents", [])

    if not isinstance(documents, list) or len(documents) == 0:
        return _error("Field 'documents' must be a non-empty list.")
    if len(documents) > MAX_BATCH:
        return _error(f"Batch size exceeds maximum of {MAX_BATCH} documents.")

    texts, doc_ids, sources = [], [], []
    for i, doc in enumerate(documents):
        text = doc.get("text", "") if isinstance(doc, dict) else doc
        err = _validate_text(text)
        if err:
            return _error(f"Document #{i}: {err}")
        texts.append(text)
        doc_ids.append(doc.get("document_id", str(uuid.uuid4())) if isinstance(doc, dict) else str(uuid.uuid4()))
        sources.append(doc.get("source") if isinstance(doc, dict) else None)

    results = _classifier().predict_batch(texts, doc_ids)

    output = []
    for result, text, source in zip(results, texts, sources):
        _persist_record(result.document_id, text, source, result)
        output.append({
            "document_id": result.document_id,
            "predicted_category": result.predicted_category,
            "confidence": result.confidence,
            "sentiment": result.sentiment,
            "probabilities": result.probabilities,
            "tokens_extracted": result.tokens_extracted,
        })

    return jsonify({
        "results": output,
        "total_processed": len(output),
    }), 200


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

@api_bp.route("/documents", methods=["GET"])
def list_documents():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    category = request.args.get("category")

    query = db.session.query(Document)
    if category:
        query = query.join(ClassificationRecord).filter(
            ClassificationRecord.predicted_category == category
        )

    paginated = query.order_by(Document.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        "documents": [d.to_dict() for d in paginated.items],
        "page": page,
        "per_page": per_page,
        "total": paginated.total,
        "pages": paginated.pages,
    })


@api_bp.route("/documents/<string:doc_id>", methods=["GET"])
def get_document(doc_id: str):
    doc = db.session.query(Document).filter_by(document_id=doc_id).first()
    if not doc:
        return _error(f"Document '{doc_id}' not found.", 404)

    latest = (
        db.session.query(ClassificationRecord)
        .filter_by(document_id=doc_id)
        .order_by(ClassificationRecord.created_at.desc())
        .first()
    )

    return jsonify({
        "document": doc.to_dict(),
        "latest_classification": latest.to_dict() if latest else None,
    })


@api_bp.route("/documents/<string:doc_id>", methods=["DELETE"])
def delete_document(doc_id: str):
    doc = db.session.query(Document).filter_by(document_id=doc_id).first()
    if not doc:
        return _error(f"Document '{doc_id}' not found.", 404)
    db.session.delete(doc)
    db.session.commit()
    return jsonify({"message": f"Document '{doc_id}' deleted."}), 200


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@api_bp.route("/stats", methods=["GET"])
def stats():
    total_docs = db.session.query(Document).count()
    total_classifications = db.session.query(ClassificationRecord).count()

    category_counts = (
        db.session.query(
            ClassificationRecord.predicted_category,
            db.func.count(ClassificationRecord.id).label("count"),
        )
        .group_by(ClassificationRecord.predicted_category)
        .all()
    )

    avg_confidence = db.session.query(
        db.func.avg(ClassificationRecord.confidence)
    ).scalar()

    return jsonify({
        "total_documents": total_docs,
        "total_classifications": total_classifications,
        "avg_confidence": round(float(avg_confidence or 0), 4),
        "category_distribution": {r.predicted_category: r.count for r in category_counts},
    })


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

def _persist_record(doc_id: str, text: str, source, result) -> None:
    try:
        if not db.session.query(Document).filter_by(document_id=doc_id).first():
            doc = Document(
                document_id=doc_id,
                content=text[:10_000],
                source=source,
                word_count=len(text.split()),
            )
            db.session.add(doc)

        record = ClassificationRecord(
            document_id=doc_id,
            predicted_category=result.predicted_category,
            confidence=result.confidence,
            sentiment_label=result.sentiment.get("label"),
            sentiment_score=result.sentiment.get("score"),
            probabilities_json=json.dumps(result.probabilities),
            processing_time_ms=result.processing_time_ms,
            tokens_extracted=result.tokens_extracted,
        )
        db.session.add(record)
        db.session.commit()
    except Exception as exc:
        logger.error("DB persist failed: %s", exc)
        db.session.rollback()
