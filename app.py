# ============================================================
#  app.py  —  Flask backend for CardioSense_Standalone.html
#  Connects:  CardioSense_Standalone.html  ↔  dsm_v13.py
#
#  USAGE:
#    Step 1:  python dsm_v13.py train          (once)
#    Step 2:  python app.py                    (start server)
#    Step 3:  Open CardioSense_Standalone.html (auto-detects backend)
# ============================================================

import os, sys, tempfile, logging
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dsm_v13 import IsotonicCalibrator

# ── dsm_v13 must be in the same folder ──
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
import librosa

from dsm_v13 import (
    extract_features,
    compute_sqi,
    _get_prob_abn,
    SAMPLE_RATE,
    LABEL_ABNORMAL,
    OUTPUT_DIR,
    ThreeWayEnsemble,   # required so joblib can unpickle pipeline.pkl
    WeightedHGB,        # required so joblib can unpickle pipeline.pkl
)

# Make custom classes visible under __main__ so pickle can resolve them
import sys as _sys
_sys.modules[__name__].ThreeWayEnsemble = ThreeWayEnsemble
_sys.modules[__name__].WeightedHGB      = WeightedHGB

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app   = Flask(__name__)
CORS(app)   # Allow requests from file:// or any origin

# ── Model cache (loaded once) ──
_pipeline   = None
_threshold  = None
_calibrator = None


def load_model():
    global _pipeline, _threshold, _calibrator
    if _pipeline is not None:
        return True
    pkl = OUTPUT_DIR / "pipeline.pkl"
    thr = OUTPUT_DIR / "threshold.pkl"
    cal = OUTPUT_DIR / "calibrator.pkl"
    if not pkl.exists():
        log.error(f"Model not found: {pkl}  →  Run: python dsm_v13.py train")
        return False
    _pipeline   = joblib.load(pkl)
    _threshold  = float(joblib.load(thr)) if thr.exists() else 0.40
    _calibrator = joblib.load(cal) if cal.exists() else None
    log.info(f"Model ready  |  threshold = {_threshold:.3f}")
    return True



def ai_interpretation(label, prob_abn, conf, sqi, hr, referral):
    """Return {summary, recommendation, sigNote} matching HTML's render()."""
    abn      = label == "Abnormal"
    sqi_word = "good" if sqi > 0.65 else "acceptable" if sqi > 0.35 else "low"
    hr_word  = ("bradycardic (slow)" if hr < 60
                else "tachycardic (fast)" if hr > 100
                else "within normal range")
    conf_pct = f"{conf*100:.0f}%"

    summary = (
        f"Automated analysis detected patterns consistent with cardiac abnormality. "
        f"Estimated heart rate is {hr:.0f} bpm ({hr_word}). "
        f"P(Abnormal) = {prob_abn*100:.1f}%."
    ) if abn else (
        f"Phonocardiogram shows characteristics consistent with normal heart sounds. "
        f"Estimated heart rate is {hr:.0f} bpm ({hr_word}). "
        f"P(Abnormal) = {prob_abn*100:.1f}%."
    )
    recommendation = (
        "⚠ Referral is recommended. Please consult a cardiologist for further "
        "evaluation including clinical auscultation and echocardiography."
    ) if abn else (
        "✅ No immediate referral required based on automated analysis. "
        "Continue routine cardiac health monitoring."
    )
    sig_note = (
        f"Signal quality index: {sqi:.2f} ({sqi_word}). "
        f"Model confidence: {conf_pct}. "
        f"DSM v13 — RobustScaler → PCA(60) → 3-model ensemble with isotonic calibration. "
        + ("Note: Low signal quality — re-recording may improve accuracy."
           if sqi < 0.30 else "Signal quality sufficient for analysis.")
    )

    return {"summary": summary, "recommendation": recommendation, "sigNote": sig_note}


# ============================================================
#  POST /api/predict
# ============================================================
@app.route("/api/predict", methods=["POST"])
def predict():
    if not load_model():
        return jsonify({"error": "Model not trained. Run: python dsm_v13.py train"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (field name: 'file')"}), 400

    wav = request.files["file"]
    if not wav.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only .wav files are supported."}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wav.save(tmp_path)

    try:
        raw, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        duration = len(raw) / SAMPLE_RATE

        sqi   = float(compute_sqi(raw))
        feats = extract_features(raw, force=True, augment=False)

        if feats is None:
            return jsonify({
                "label":      "Unsure",
                "probAbn":    0.5,
                "confidence": 0.0,
                "threshold":  _threshold,
                "sqi":        round(sqi, 4),
                "referral":   True,
                "hr":         75.0,
                "duration":   round(duration, 2),
                "source":     "dsm_v13",
                "ai": {
                    "summary":        "Feature extraction failed — recording may be too short or noisy.",
                    "recommendation": "Try a cleaner recording of at least 3 seconds.",
                    "sigNote":        f"Signal quality: {sqi:.2f}. Minimum 3 s required.",
                },
            })

        X        = np.array([list(feats.values())], dtype=np.float32)
        prob_abn = float(_get_prob_abn(_pipeline, X, calibrator=_calibrator)[0])

        label    = "Abnormal" if prob_abn >= _threshold else "Normal"
        conf     = prob_abn if label == "Abnormal" else 1.0 - prob_abn
        referral = label == "Abnormal"

        # Estimate heart rate from features if available
        hr = float(feats.get("hrv_mean_rr", 0))
        if hr > 0:
            hr = min(200, max(30, 60000 / hr))   # ms → bpm
        else:
            hr = 75.0

        ai = ai_interpretation(label, prob_abn, conf, sqi, hr, referral)

        return jsonify({
            "label":      label,
            "probAbn":    round(prob_abn, 4),     # matches HTML's result.probAbn
            "confidence": round(conf, 4),
            "threshold":  round(_threshold, 4),
            "sqi":        round(sqi, 4),
            "referral":   referral,
            "hr":         round(hr, 1),
            "duration":   round(duration, 2),
            "source":     "dsm_v13",              # tells HTML this is the real model
            "ai":         ai,
        })

    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        os.unlink(tmp_path)


# ── Health check (HTML pings this to detect backend) ──
@app.route("/api/health", methods=["GET"])
def health():
    ready = (OUTPUT_DIR / "pipeline.pkl").exists()
    return jsonify({
        "status":      "ok",
        "model_ready": ready,
        "version":     "DSM v13",
    })


if __name__ == "__main__":
    log.info("CardioSense backend starting → http://localhost:5000")
    log.info("Ensure model is trained: python dsm_v13.py train")
    app.run(host="0.0.0.0", port=5000, debug=False)
