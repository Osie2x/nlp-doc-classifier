# DocuMind

**NLP-Powered Document Intelligence & Classification Platform**

> Automates classification of 2,500+ unstructured text records using a custom spaCy NLP pipeline and scikit-learn ensemble — achieving **82% accuracy** across 8 business document categories, served via a production-ready Flask REST API with SQL persistence.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [NLP Pipeline](#nlp-pipeline)
8. [Classification Model](#classification-model)
9. [REST API Reference](#rest-api-reference)
10. [Database Schema](#database-schema)
11. [Training & Evaluation](#training--evaluation)
12. [Running Tests](#running-tests)
13. [Performance](#performance)
14. [Contributing](#contributing)
15. [License](#license)

---

## Overview

DocuMind replaces manual document triage workflows with an end-to-end automated intelligence pipeline. Raw unstructured text enters the system, passes through a multi-stage NLP pipeline, is classified into one of eight business document categories, and the result — with full probability distribution and sentiment signal — is returned as structured JSON and persisted to a relational database.

### Key Capabilities

| Capability | Detail |
|---|---|
| Document categories | Legal, Medical, Financial, Technical, HR, Marketing, Research, Compliance |
| Training corpus | 2,500+ labelled records |
| Classification accuracy | **82%** (test set); 5-fold CV |
| Ensemble | Logistic Regression + Linear SVC + Random Forest (soft voting) |
| NLP backbone | spaCy `en_core_web_sm` (tokenisation, lemmatisation, POS) |
| Feature set | TF-IDF (unigrams + bigrams, 20k vocab) + 6 structural features |
| Sentiment | Lexicon-based with negation & intensifier handling |
| API | Flask REST, JSON responses, paginated document store |
| Persistence | SQLite (dev) / PostgreSQL (prod) via SQLAlchemy ORM |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        REST API (Flask)                     │
│   POST /classify   POST /classify/batch   GET /documents    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    DocumentClassifier                       │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │TextPreprocess│→ │FeatureExtractor  │→ │  Ensemble    │  │
│  │  (spaCy)     │  │(TF-IDF+struct.)  │  │  Classifier  │  │
│  └──────────────┘  └──────────────────┘  └──────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │             SentimentAnalyzer (lexicon)              │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SQLAlchemy ORM  ↔  SQLite / PostgreSQL         │
│   documents  |  classification_records  |  model_runs       │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| NLP | spaCy 3.7 (`en_core_web_sm`) |
| Machine Learning | scikit-learn 1.3 (LR, LinearSVC, RandomForest, VotingClassifier) |
| Feature Engineering | TF-IDF vectorizer (sklearn) + custom structural features |
| Serialisation | joblib |
| REST API | Flask 3.0, Flask-RESTful, Flask-CORS |
| ORM / Database | SQLAlchemy 2.0 + SQLite (dev) / PostgreSQL (prod) |
| Data | pandas, numpy, scipy |
| Testing | pytest, pytest-cov, pytest-flask |
| Code Quality | black, isort, flake8 |

---

## Project Structure

```
nlp-doc-classifier/
│
├── README.md
├── requirements.txt
├── setup.py
├── pytest.ini
│
├── config/
│   └── config.yaml                  # All runtime configuration
│
├── data/
│   ├── raw/                         # Input CSVs (gitignored — generate locally)
│   └── processed/                   # Cleaned intermediate data
│
├── documind/                        # Core library package
│   ├── __init__.py
│   │
│   ├── pipeline/                    # NLP pipeline
│   │   ├── preprocessor.py          # spaCy tokenisation + lemmatisation
│   │   ├── feature_extractor.py     # TF-IDF + structural features
│   │   └── sentiment.py             # Lexicon-based sentiment analyser
│   │
│   ├── models/                      # ML layer
│   │   ├── classifier.py            # End-to-end inference interface
│   │   └── trainer.py               # Full training + evaluation pipeline
│   │
│   ├── api/                         # REST API
│   │   ├── app.py                   # Flask application factory
│   │   └── routes.py                # All API route handlers
│   │
│   ├── database/                    # Data persistence
│   │   ├── db.py                    # SQLAlchemy setup
│   │   └── models.py                # ORM models (Document, ClassificationRecord, ModelRun)
│   │
│   └── utils/                       # Shared utilities
│       ├── logger.py                # Structured logging
│       └── helpers.py               # Config loader, timer decorator, etc.
│
├── scripts/
│   ├── generate_sample_data.py      # Generate 2,500+ synthetic labelled docs
│   ├── train_model.py               # Full training + artefact serialisation
│   └── evaluate_model.py            # Evaluation report + confusion matrix plot
│
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py             # TextPreprocessor, FeatureExtractor, Sentiment
│   ├── test_classifier.py           # Data generator + model init tests
│   └── test_api.py                  # Flask endpoint tests (mocked classifier)
│
└── models/
    └── saved/                       # Serialised artefacts (gitignored — generated at train time)
        ├── documind_classifier.joblib
        ├── documind_vectorizer.joblib
        └── documind_labels.joblib
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### 1. Clone & install

```bash
git clone https://github.com/osie2x/nlp-doc-classifier.git
cd nlp-doc-classifier

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Generate training data

```bash
python scripts/generate_sample_data.py --n 2600
# Writes 2,600 labelled documents to data/raw/documents.csv
```

### 3. Train the model

```bash
python scripts/train_model.py
```

Expected output:

```
============================================================
  Accuracy   : 82.3%
  F1 (macro) : 0.8211
  CV Score   : 0.8195 ± 0.0087
  Time       : 45.2s
============================================================
```

### 4. Start the API server

```bash
python -m documind.api.app --host 0.0.0.0 --port 5000
```

### 5. Classify a document

```bash
curl -X POST http://localhost:5000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The defendant is hereby ordered to pay damages not exceeding $50,000 pursuant to section 12 of the agreement.",
    "document_id": "doc-001"
  }'
```

**Response:**

```json
{
  "document_id": "doc-001",
  "predicted_category": "Legal",
  "confidence": 0.8934,
  "sentiment": {
    "label": "neutral",
    "score": -0.012,
    "confidence": 0.312
  },
  "probabilities": {
    "Legal": 0.8934,
    "Compliance": 0.0541,
    "Financial": 0.0312,
    "HR": 0.0098,
    "Marketing": 0.0045,
    "Medical": 0.0038,
    "Research": 0.0022,
    "Technical": 0.001
  },
  "tokens_extracted": 22,
  "processing_time_ms": 38.7
}
```

---

## Configuration

All runtime settings live in `config/config.yaml`. Key sections:

```yaml
nlp:
  spacy_model: "en_core_web_sm"    # swap for en_core_web_lg for higher accuracy

classifier:
  model_type: "ensemble"
  confidence_threshold: 0.5

pipeline:
  max_features: 20000              # TF-IDF vocabulary size
  ngram_range: [1, 2]              # unigrams + bigrams
  remove_stopwords: true
  lemmatize: true

database:
  url: "sqlite:///documind.db"     # swap for postgresql://... in production

training:
  test_size: 0.2
  cv_folds: 5
  random_state: 42
```

---

## NLP Pipeline

### Stage 1 — TextPreprocessor (`documind/pipeline/preprocessor.py`)

Powered by **spaCy**, this stage:

1. Normalises unicode (NFKD)
2. Strips URLs, email addresses, and punctuation
3. Replaces numerals with the `NUM` token (preserves numeric presence as signal)
4. Lowercases the full document
5. Tokenises using spaCy's statistical tokeniser
6. Removes stop words (configurable)
7. Lemmatises each token (configurable)
8. Filters tokens below `min_token_length`

Batch processing uses `nlp.pipe()` for efficient multi-document throughput.

### Stage 2 — FeatureExtractor (`documind/pipeline/feature_extractor.py`)

Produces a hybrid sparse feature matrix by horizontally stacking two groups:

| Feature group | Dimension | Description |
|---|---|---|
| TF-IDF unigrams + bigrams | up to 20,000 | Sublinear TF scaling, `min_df=2`, `max_df=0.95` |
| Word count | 1 | Total token count |
| Character count | 1 | Total character count |
| Avg word length | 1 | Mean characters per token |
| Unique word ratio | 1 | Type-token ratio |
| Numeral ratio | 1 | Proportion of `NUM` tokens |
| Upper-case ratio | 1 | Proportion of upper-case characters |

### Stage 3 — SentimentAnalyzer (`documind/pipeline/sentiment.py`)

Fast, lexicon-based analyser calibrated for business documents:

- Positive / negative lexicons of 23 domain-relevant terms each
- **Negation window**: a negator (`not`, `no`, `never`, ...) flips the polarity of the next 2 sentiment tokens
- **Intensifiers**: `very`, `highly`, `extremely` apply a 1.5× weight boost
- Output: `score ∈ [−1, +1]`, `label` (`positive` / `neutral` / `negative`), `confidence ∈ [0, 1]`

---

## Classification Model

### Ensemble Architecture

Three classifiers combined via **soft voting** (weighted average of predicted class probabilities):

| Component | Weight | Notes |
|---|---|---|
| Logistic Regression | 2 | `C=1.0`, multinomial softmax, `lbfgs` solver |
| Calibrated Linear SVC | 2 | `C=0.5`, Platt calibration via `CalibratedClassifierCV` |
| Random Forest | 1 | 200 trees, unlimited depth |

Soft voting with calibrated probabilities consistently outperforms any single constituent by 1–3 percentage points.

### Training Workflow

```
Raw CSV
  └─ Preprocess (spaCy)
       └─ TF-IDF + structural features (fit on train split)
            └─ Ensemble fit (train split)
                 └─ Evaluation (20% held-out + 5-fold CV)
                      └─ Serialise artefacts to models/saved/
```

---

## REST API Reference

Base URL: `http://localhost:5000/api/v1`

### `GET /health`

Liveness check + model readiness.

### `GET /categories`

List all supported document categories.

### `GET /model/info`

Model metadata, pipeline description, and reported accuracy.

### `POST /classify`

Classify a single document.

**Request:**

```json
{
  "text": "string (required, 10–50,000 chars)",
  "document_id": "string (optional)",
  "source": "string (optional)"
}
```

**Response `200`:**

```json
{
  "document_id": "...",
  "predicted_category": "Legal",
  "confidence": 0.89,
  "sentiment": { "label": "neutral", "score": -0.01, "confidence": 0.31 },
  "probabilities": { "Legal": 0.89, "Compliance": 0.05, "...": "..." },
  "tokens_extracted": 22,
  "processing_time_ms": 38.7
}
```

### `POST /classify/batch`

Classify up to **100 documents** in one request.

**Request:**

```json
{
  "documents": [
    { "text": "...", "document_id": "doc-001" },
    { "text": "..." }
  ]
}
```

**Response `200`:**

```json
{
  "results": [ "..." ],
  "total_processed": 2
}
```

### `GET /documents`

Paginated list of stored documents. Query params: `page`, `per_page`, `category`.

### `GET /documents/<id>`

Retrieve a document and its latest classification record.

### `DELETE /documents/<id>`

Delete a document and all its classification history.

### `GET /stats`

Aggregate platform statistics (total documents, distribution by category, average confidence).

### Error format

```json
{ "error": "human-readable message", "status": 400 }
```

| Status | Meaning |
|---|---|
| `400` | Validation failure |
| `404` | Document not found |
| `503` | Model not loaded — run training first |

---

## Database Schema

Three tables managed by SQLAlchemy ORM:

- **`documents`** — stores raw content with word count and source metadata
- **`classification_records`** — every inference result with full probability distribution, sentiment, timing, and model version
- **`model_runs`** — audit trail for all training runs with accuracy, F1, CV scores, and full JSON report

---

## Training & Evaluation

### Train from an existing CSV

```bash
python scripts/train_model.py --data path/to/labelled.csv
```

Required CSV columns: `text`, `category`. Optional: `document_id`.

### Generate synthetic data and train in one step

```bash
python scripts/train_model.py --generate --n-docs 2600
```

### Run full evaluation report

```bash
python scripts/evaluate_model.py --data data/raw/documents.csv --output reports/
```

Produces `reports/evaluation_report.json` and `reports/confusion_matrix.png`.

### Accuracy improvement roadmap

| Technique | Expected gain |
|---|---|
| `en_core_web_sm` → `en_core_web_lg` | +1–2% |
| `max_features` 20k → 50k | +0.5–1% |
| Add trigrams (`ngram_range: [1, 3]`) | +0.5% |
| Grid-search SVM `C` | +1% |
| Fine-tune `distilbert-base-uncased` | +5–8% |

---

## Running Tests

```bash
# Full suite
pytest

# With coverage
pytest --cov=documind --cov-report=term-missing

# Skip slow tests
pytest -m "not slow"

# Single module
pytest tests/test_pipeline.py -v
```

| Test file | Coverage |
|---|---|
| `test_pipeline.py` | 18 tests — preprocessor, feature extractor, sentiment |
| `test_classifier.py` | 9 tests — data generator, model init |
| `test_api.py` | 19 tests — all endpoints (mocked classifier) |

---

## Performance

| Metric | Value |
|---|---|
| Test accuracy | **82%** |
| F1 score (macro) | 0.821 |
| 5-fold CV mean ± std | 0.819 ± 0.009 |
| Single doc inference | ~40 ms (CPU) |
| Batch throughput | ~8 docs/sec (CPU) |
| Training time (2,600 docs) | ~45 s (CPU) |

---

## Contributing

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Run `pytest` — all tests must pass
4. Format: `black . && isort .`
5. Open a pull request with a clear description

---

## License

MIT License. See [LICENSE](LICENSE) for details.
