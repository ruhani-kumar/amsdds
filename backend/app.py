from __future__ import annotations

import os

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from backend.config import AppConfig, load_config
from backend.context import set_config
from backend.routes import health_bp, model_info_bp, predict_bp

# 16 MB upload ceiling - generous for a lesion photo, stops obvious abuse.
MAX_CONTENT_LENGTH = 16 * 1024 * 1024


def create_app(config: AppConfig | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    set_config(app, config or load_config())

    app.register_blueprint(health_bp)
    app.register_blueprint(model_info_bp)
    app.register_blueprint(predict_bp)

    _register_error_handlers(app)
    return app


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(400)
    def _bad_request(err):  # noqa: ANN001
        return jsonify({"success": False, "error": _msg(err, "Bad request.")}), 400

    @app.errorhandler(404)
    def _not_found(err):  # noqa: ANN001
        return jsonify({"success": False, "error": "Not found."}), 404

    @app.errorhandler(413)
    def _too_large(err):  # noqa: ANN001
        return jsonify(
            {"success": False, "error": "Uploaded file is too large (max 16 MB)."}
        ), 413

    @app.errorhandler(405)
    def _method(err):  # noqa: ANN001
        return jsonify({"success": False, "error": "Method not allowed."}), 405

    @app.errorhandler(Exception)
    def _unhandled(err):  # noqa: ANN001
        if isinstance(err, HTTPException):
            return jsonify(
                {"success": False, "error": _msg(err, "Request failed.")}
            ), err.code
        # Never leak a stack trace into the API response.
        app.logger.exception("Unhandled error")
        return jsonify(
            {"success": False, "error": "Internal server error."}
        ), 500


def _msg(err, fallback: str) -> str:
    return getattr(err, "description", None) or fallback


# Module-level app for `flask --app backend.app run` / WSGI servers.
# Only reads configs/thresholds.yaml; does not load the model checkpoint.
app = create_app()


def main() -> None:
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("BACKEND_PORT", "5000")))  
    app.run(host=host, port=port, debug=_debug())


def _debug() -> bool:
    return os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    main()
