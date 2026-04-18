"""
train_model.py
--------------
End-to-end training script.  Reads the labelled CSV, trains the ensemble,
and serialises all artefacts to models/saved/.

Usage
-----
    python scripts/train_model.py
    python scripts/train_model.py --data data/raw/documents.csv --model-dir models/saved
    python scripts/train_model.py --generate   # create synthetic data first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from documind.models import ModelTrainer
from documind.utils import ensure_dir, get_logger

logger = get_logger("train")


def load_data(path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    required = {"text", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")

    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 20]

    logger.info("Loaded %d documents across %d categories", len(df), df["category"].nunique())
    return df["text"].tolist(), df["category"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Train DocuMind document classifier")
    parser.add_argument("--data", default="data/raw/documents.csv", help="Path to labelled CSV")
    parser.add_argument("--model-dir", default="models/saved", help="Output directory for artefacts")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data first")
    parser.add_argument("--n-docs", type=int, default=2600, help="Docs to generate if --generate")
    args = parser.parse_args()

    if args.generate or not Path(args.data).exists():
        logger.info("Generating synthetic dataset (%d docs) …", args.n_docs)
        from scripts.generate_sample_data import generate_dataset, save_csv
        records = generate_dataset(n=args.n_docs)
        save_csv(records, args.data)

    texts, labels = load_data(args.data)

    trainer = ModelTrainer(
        spacy_model=args.spacy_model,
        max_features=args.max_features,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        model_dir=args.model_dir,
    )

    start = time.perf_counter()
    metrics = trainer.train(texts, labels, evaluate=True)
    elapsed = time.perf_counter() - start

    ensure_dir("models/saved")
    report_path = Path("models/saved/training_report.json")
    metrics["training_time_s"] = round(elapsed, 2)
    metrics["run_id"] = str(uuid.uuid4())
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training report saved to %s", report_path)
    print("\n" + "=" * 60)
    print(f"  Accuracy   : {metrics['accuracy'] * 100:.1f}%")
    print(f"  F1 (macro) : {metrics['f1_macro']:.4f}")
    print(f"  CV Score   : {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"  Time       : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
