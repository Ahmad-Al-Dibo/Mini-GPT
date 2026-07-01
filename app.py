import os

from flask import Flask, jsonify, request
from arclm import load_model

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/arclm.pth")
_model = None


def get_model():
    global _model

    if _model is None:
        print(f"Loading model: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
        print("Model loaded.")

    return _model


def bounded_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default

    return max(minimum, min(value, maximum))


def bounded_float(value, default, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default

    return max(minimum, min(value, maximum))


@app.route("/")
def home():
    return jsonify({
        "name": "ArcLM Prediction API",
        "status": "running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True) or {}

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "success": False,
                "error": "Prompt is required."
            }), 400

        temperature = bounded_float(
            data.get("temperature"),
            0.8,
            0.1,
            2.0,
        )

        top_p = bounded_float(
            data.get("top_p"),
            0.9,
            0.05,
            1.0,
        )

        repetition_penalty = bounded_float(
            data.get("repetition_penalty"),
            1.1,
            1.0,
            3.0,
        )

        model = get_model()

        # Generate ONLY ONE new token
        token = model.predict(
            prompt,
            max_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return jsonify({
            "success": True,
            "token": token,
            "model_path": MODEL_PATH
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generates up to max_tokens by repeatedly predicting one token.
    Useful if Unity wants one request instead of many.
    """

    try:
        data = request.get_json(silent=True) or {}

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "success": False,
                "error": "Prompt is required."
            }), 400

        max_tokens = bounded_int(
            data.get("max_tokens"),
            100,
            1,
            1000,
        )

        temperature = bounded_float(
            data.get("temperature"),
            0.8,
            0.1,
            2.0,
        )

        top_p = bounded_float(
            data.get("top_p"),
            0.9,
            0.05,
            1.0,
        )

        repetition_penalty = bounded_float(
            data.get("repetition_penalty"),
            1.1,
            1.0,
            3.0,
        )

        model = get_model()

        text = prompt

        for _ in range(max_tokens):

            token = model.predict(
                text,
                max_new_tokens=1,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if not token:
                break

            text += token

        return jsonify({
            "success": True,
            "prompt": prompt,
            "result": text,
            "generated": text[len(prompt):],
            "model_path": MODEL_PATH
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=True,
    )