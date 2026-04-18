"""
evaluate_model.py
-----------------
Load a trained model and produce a full evaluation report including
per-class metrics, confusion matrix, and a visual heatmap.

Usage
-----
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --data data/raw/documents.csv --output reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from documind.pipeline import FeatureExtractor, TextPreprocessor
from documind.utils import ensure_dir, get_logger

logger = get_logger("evaluate")


def load_artefacts(model_dir: str):
    import joblib
    mdir = Path(model_dir)
    model = joblib.load(mdir / "documind_classifier.joblib")
    vectorizer = joblib.load(mdir / "documind_vectorizer.joblib")
    label_encoder = joblib.load(mdir / "documind_labels.joblib")
    return model, vectorizer, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocuMind classifier")
    parser.add_argument("--data", default="data/raw/documents.csv")
    parser.add_argument("--model-dir", default="models/saved")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split

    logger.info("Loading data from %s …", args.data)
    df = pd.read_csv(args.data).dropna(subset=["text", "category"])
    texts = df["text"].astype(str).tolist()
    labels = df["category"].tolist()

    logger.info("Loading artefacts from %s …", args.model_dir)
    model, vectorizer, label_encoder = load_artefacts(args.model_dir)

    preprocessor = TextPreprocessor(spacy_model=args.spacy_model)
    extractor = FeatureExtractor()
    extractor.vectorizer = vectorizer
    extractor._is_fitted = True

    _, X_test_raw, _, y_test_raw = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info("Preprocessing %d test documents …", len(X_test_raw))
    processed = preprocessor.preprocess_batch(X_test_raw)
    features = extractor.transform(processed)

    y_test = label_encoder.transform(y_test_raw)
    y_pred = model.predict(features)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    ensure_dir(args.output)
    out = Path(args.output)

    report_path = out / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "accuracy": round(float(acc), 4),
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "categories": list(label_encoder.classes_),
            },
            f,
            indent=2,
        )

    _save_confusion_matrix_plot(cm, label_encoder.classes_, out / "confusion_matrix.png")

    print("\n" + "=" * 60)
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Report saved  : {report_path}")
    print("=" * 60)
    print("\nPer-class F1 scores:")
    for cat in label_encoder.classes_:
        r = report.get(cat, {})
        print(f"  {cat:<15} F1={r.get('f1-score', 0):.3f}  "
              f"P={r.get('precision', 0):.3f}  R={r.get('recall', 0):.3f}")


def _save_confusion_matrix_plot(cm, classes, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("DocuMind — Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Confusion matrix plot saved to %s", path)
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping plot")


if __name__ == "__main__":
    main()
