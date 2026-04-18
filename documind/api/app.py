"""Flask application factory."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from flask import Flask
from flask_cors import CORS

from documind.database import init_db
from documind.models import DocumentClassifier
from documind.utils import get_logger

logger = get_logger(__name__)


def create_app(config_path: str = "config/config.yaml") -> Flask:
    app = Flask(__name__, instance_relative_config=False)

    cfg = _load_config(config_path)

    app.config["SECRET_KEY"] = cfg["app"].get("secret_key", "dev-secret-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = cfg["database"]["url"]
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["DEBUG"] = cfg["app"].get("debug", False)
    app.config["DOCUMIND_CONFIG"] = cfg

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    init_db(app)

    classifier = DocumentClassifier(
        model_path=cfg["classifier"]["model_path"],
        vectorizer_path=cfg["classifier"]["vectorizer_path"],
        label_encoder_path=cfg["classifier"]["label_encoder_path"],
        spacy_model=cfg["nlp"]["spacy_model"],
        confidence_threshold=cfg["classifier"]["confidence_threshold"],
    )
    app.config["CLASSIFIER"] = classifier

    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    logger.info(
        "DocuMind API ready — model ready: %s", classifier.is_ready()
    )
    return app


def _load_config(path: str) -> dict:
    config_file = Path(path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(config_file) as f:
        return yaml.safe_load(f)


def main():
    """CLI entry point — python -m documind.api.app or documind-serve."""
    import click

    @click.command()
    @click.option("--host", default="0.0.0.0", help="Bind host")
    @click.option("--port", default=5000, type=int, help="Bind port")
    @click.option("--config", default="config/config.yaml", help="Config file")
    @click.option("--debug", is_flag=True, default=False)
    def run(host, port, config, debug):
        app = create_app(config)
        app.run(host=host, port=port, debug=debug)

    run()


if __name__ == "__main__":
    main()
