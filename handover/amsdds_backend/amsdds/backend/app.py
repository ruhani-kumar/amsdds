import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from engine import Engine

app = Flask(__name__)
CORS(app)
engine = Engine(os.environ.get("AMSDDS_CONFIG", "config/thresholds.yaml"))

@app.get("/health")
def health(): return jsonify(status="ok", device=engine.device)

@app.get("/model-info")
def model_info(): return jsonify(engine.model_info())

@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None: return jsonify(error="multipart field 'image' required"), 400
    try:
        return jsonify(engine.predict(
            f.read(),
            age=request.form.get("age"),
            sex=request.form.get("sex"),
            localization=request.form.get("localization"),
            layer2_head=request.form.get("layer2_head")))
    except Exception as e:                                     # noqa: BLE001
        return jsonify(error=f"inference failed: {type(e).__name__}: {e}"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=False)
