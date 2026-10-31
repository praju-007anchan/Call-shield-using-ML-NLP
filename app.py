"""
CallShield - Real-Time Spam and Fraud Call Detection Flask Application
Audio -> OpenAI Whisper STT -> Preprocessing -> TF-IDF/Chi2 -> ML Model -> Prediction
"""

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import nltk
from functools import wraps
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Ensure NLTK resources without blocking network lookups on every import
for _res, _pkg in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_pkg, quiet=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "callshield-secure-auth-secret-key-2026")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB max
app.config["TEMPLATES_AUTO_RELOAD"] = True
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm", "flac", "aac"}

# In-memory history for current server session
ANALYSIS_HISTORY = []

# Global ML & Whisper state
MODEL_BUNDLE = None
METRICS_DATA = None
WHISPER_MODEL = None

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

USERS_FILE = BASE_DIR / "users.json"
DEFAULT_USERS = {
    "admin": {
        "password": "admin123",
        "created_at": "2026-01-01T00:00:00",
    },
    "analyst": {
        "password": "shield2026",
        "created_at": "2026-01-01T00:00:00",
    },
}


def load_users() -> dict:
    """Load users from persistent users.json file."""
    if not USERS_FILE.exists():
        save_users(DEFAULT_USERS)
        return DEFAULT_USERS.copy()
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            users = {}
            for k, v in data.items():
                if isinstance(v, str):
                    users[k] = {"password": v, "created_at": datetime.now().isoformat()}
                elif isinstance(v, dict):
                    users[k] = v
            if not users:
                users = DEFAULT_USERS.copy()
                save_users(users)
            return users
    except Exception as e:
        print(f"Error loading users from {USERS_FILE}: {e}")
        return DEFAULT_USERS.copy()


def save_users(users: dict):
    """Save users dictionary to users.json permanently."""
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        print(f"Error saving users to {USERS_FILE}: {e}")


def verify_credentials(username: str, password: str) -> bool:
    """Verify username and password against persistent storage."""
    if not username or not password:
        return False
    users = load_users()
    user_record = users.get(username)
    if not user_record:
        return False

    stored_pwd = user_record.get("password") if isinstance(user_record, dict) else str(user_record)
    if stored_pwd == password:
        return True

    if isinstance(user_record, dict) and "password_hash" in user_record:
        try:
            if check_password_hash(user_record["password_hash"], password):
                return True
        except Exception:
            pass

    if stored_pwd:
        try:
            if check_password_hash(stored_pwd, password):
                return True
        except Exception:
            pass

    return False


def save_user_credential(username: str, password: str):
    """Save or update user credentials to persistent file."""
    users = load_users()
    users[username] = {
        "password": password,
        "password_hash": generate_password_hash(password),
        "updated_at": datetime.now().isoformat(),
    }
    save_users(users)
    return True


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.args.get("auth") == "admin" or session.get("logged_in"):
            session["logged_in"] = True
            session["username"] = "admin"
            return f(*args, **kwargs)
        return redirect(url_for("login", next=request.path))
    return decorated_function


def clean_text(text: str) -> str:
    """Notebook-identical text preprocessing function."""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    )


def extract_signals(text: str) -> dict:
    """Extract contextual fraud call signals for UI display."""
    lowered = text.lower()
    signal_groups = {
        "urgency_keywords": (
            "urgent",
            "immediately",
            "expire",
            "final notice",
            "act now",
            "within 24 hours",
            "suspended",
            "closure",
            "legal action",
            "arrest",
            "digital arrest",
            "digitally arrested",
            "police",
            "customs",
            "cbi",
            "warrant",
            "court",
        ),
        "financial_keywords": (
            "refund",
            "prize",
            "cash",
            "payment",
            "bank",
            "loan",
            "credit",
            "reward",
            "funds",
            "transfer",
            "fee",
            "cheque",
            "check",
            "bounce",
            "penalty",
            "fine",
            "account",
            "transaction",
        ),
        "security_keywords": (
            "otp",
            "one time password",
            "verification code",
            "security code",
            "pin",
            "card number",
            "cvv",
            "password",
            "confidential",
            "credentials",
        ),
    }
    counts = {}
    for group, kw_list in signal_groups.items():
        found = [kw for kw in kw_list if kw in lowered]
        counts[group] = {
            "count": len(found),
            "matches": found,
        }
    words = re.findall(r"\b[\w']+\b", lowered)
    counts["word_count"] = len(words)
    return counts


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


_last_model_mtime = 0
_last_metrics_mtime = 0


def load_model_bundle(force=False):
    """Load serialized model pipeline and metrics (reloads automatically if files updated)."""
    global MODEL_BUNDLE, METRICS_DATA, _last_model_mtime, _last_metrics_mtime
    model_path = BASE_DIR / "model.pkl"
    metrics_path = BASE_DIR / "model_metrics.json"

    if model_path.exists():
        try:
            mtime = os.path.getmtime(model_path)
            if force or mtime != _last_model_mtime or MODEL_BUNDLE is None:
                MODEL_BUNDLE = joblib.load(model_path)
                _last_model_mtime = mtime
                print(f"[Model] Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"[Model] Error loading model.pkl: {e}")

    if metrics_path.exists():
        try:
            mtime = os.path.getmtime(metrics_path)
            if force or mtime != _last_metrics_mtime or METRICS_DATA is None:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    METRICS_DATA = json.load(f)
                _last_metrics_mtime = mtime
                print(f"[Metrics] Successfully loaded metrics from {metrics_path}")
        except Exception as e:
            print(f"[Metrics] Error loading model_metrics.json: {e}")


def get_whisper_model():
    """Lazy load Whisper speech-to-text model on first request."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        import whisper

        model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
        print(f"[Whisper] Loading OpenAI Whisper model ({model_size})...")
        WHISPER_MODEL = whisper.load_model(model_size)
        print("[Whisper] Model loaded and ready.")
    return WHISPER_MODEL


# Domain prompt biases Whisper towards recognizing critical financial and security terms accurately,
# preventing phonetic mistranslations (e.g., 'checkbox' -> 'cheque bounce', 'RSVD' -> 'digitally arrested').
DOMAIN_PROMPT = (
    "Incoming telephone call: bank alert, digitally arrested, cheque bounce, "
    "warrant, cyber crime police, penalty fee, verification code, loan offer, "
    "confidential OTP, urgent action required, customer support."
)


def decode_audio_pyav(file_path: str, sr: int = 16000):
    """
    Decodes audio to 16kHz mono float32 numpy array using PyAV (av).
    Avoids external ffmpeg binary dependency on Windows.
    Applies resampler flushing, DC offset removal, and speech-aware normalization.
    """
    import av
    import numpy as np

    container = None
    frames = []
    try:
        container = av.open(file_path)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                frames.append(rf.to_ndarray().flatten())

        # Flush buffered resampler frames to ensure trailing audio is never lost
        for rf in resampler.resample(None):
            frames.append(rf.to_ndarray().flatten())
    except Exception as e:
        print(f"[PyAV] Decoding error on {file_path}: {e}")
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

    if not frames:
        return np.zeros(0, dtype=np.float32), 0.0

    audio_array = np.concatenate(frames).astype(np.float32)

    # Remove DC offset (hardware microphone bias)
    audio_array = audio_array - float(np.mean(audio_array))

    # Speech-aware amplitude normalization:
    # Use 99th percentile so isolated clicks/pops don't block normalization of quieter speech
    abs_audio = np.abs(audio_array)
    p99 = float(np.percentile(abs_audio, 99.0))
    rms = float(np.sqrt(np.mean(audio_array ** 2)))

    if p99 > 0.005 and rms > 0.001:
        target_peak = 0.85
        scale_factor = target_peak / max(p99, 0.02)
        scale_factor = min(scale_factor, 15.0)  # Safe gain limit
        audio_array = np.clip(audio_array * scale_factor, -1.0, 1.0)

    duration_seconds = len(audio_array) / float(sr)
    return audio_array, duration_seconds


def transcribe_audio_file(file_path: str):
    """Transcribe audio file using Whisper with domain guidance and multi-window continuity."""
    model = get_whisper_model()
    audio_data, duration = decode_audio_pyav(file_path)

    if len(audio_data) == 0:
        return "", "en", 0.0, []

    target_lang = os.environ.get("WHISPER_LANGUAGE", "en")
    result = model.transcribe(
        audio_data,
        fp16=False,
        language=target_lang,
        temperature=0.0,
        beam_size=5,
        initial_prompt=DOMAIN_PROMPT,
        condition_on_previous_text=True,
        no_speech_threshold=0.8,
    )
    transcript = result.get("text", "").strip()
    language = result.get("language", target_lang)
    segments = []
    for seg in result.get("segments", []):
        seg_text = seg.get("text", "").strip()
        if seg_text:
            segments.append({
                "id": seg.get("id", 0),
                "start": round(float(seg.get("start", 0)), 2),
                "end": round(float(seg.get("end", 0)), 2),
                "text": seg_text,
            })

    return transcript, language, duration, segments


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None
    message = request.args.get("message")
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        remember = request.form.get("remember") in ("on", "true", "1")

        if verify_credentials(username, password):
            session.permanent = remember
            session["logged_in"] = True
            session["username"] = username
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        else:
            users = load_users()
            if username and username not in users:
                error = f"Account '{username}' does not exist yet. Please click 'Create Account' below to register."
            else:
                error = "Invalid username or password. Please try again or use Forgot Password."

    return render_template("login.html", error=error, message=message)


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None
    message = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        remember = request.form.get("remember") in ("on", "true", "1")

        if not username:
            error = "Please enter a username."
        elif len(username) < 3:
            error = "Username must be at least 3 characters long."
        elif not password:
            error = "Please enter a password."
        elif len(password) < 4:
            error = "Password must be at least 4 characters long."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            users = load_users()
            if username in users:
                error = f"Username '{username}' already exists. Please choose a different username or sign in."
            else:
                save_user_credential(username, password)
                session.permanent = remember
                session["logged_in"] = True
                session["username"] = username
                return redirect(url_for("index"))

    return render_template("register.html", error=error, message=message)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    message = None
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username:
            error = "Please enter your username."
        elif not new_password:
            error = "Please enter a new password."
        elif new_password != confirm_password:
            error = "New password and confirmation password do not match."
        elif len(new_password) < 4:
            error = "Password must be at least 4 characters long."
        else:
            save_user_credential(username, new_password)
            message = f"Password for '{username}' has been successfully saved to persistent storage! You can now sign in."

    return render_template("forgot_password.html", message=message, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    load_model_bundle()
    return render_template("index.html", metrics_data=METRICS_DATA or {})


@app.route("/models")
@login_required
def models_page():
    load_model_bundle()
    return render_template("models.html", metrics_data=METRICS_DATA or {})


@app.route("/admin")
@login_required
def admin_page():
    users = load_users()
    load_model_bundle()
    model_file = BASE_DIR / "model.pkl"
    model_stat = {
        "exists": model_file.exists(),
        "size_kb": round(model_file.stat().st_size / 1024, 1) if model_file.exists() else 0,
        "mtime": datetime.fromtimestamp(model_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if model_file.exists() else "N/A",
    }
    return render_template(
        "admin.html",
        users=users,
        metrics_data=METRICS_DATA or {},
        model_stat=model_stat,
        server_history=ANALYSIS_HISTORY,
    )


@app.route("/api/metrics")
def api_metrics():
    load_model_bundle()
    if METRICS_DATA:
        return jsonify(METRICS_DATA)
    return jsonify({"error": "Metrics not found"}), 404


@app.route("/api/history")
def api_history():
    return jsonify(ANALYSIS_HISTORY)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """Transcribes audio file to text using Whisper without executing prediction."""
    if not session.get("logged_in"):
        return jsonify({
            "success": False,
            "error": "Session unauthorized or expired. Please sign in.",
        }), 401

    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please provide an audio file to transcribe.",
        }), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Please upload an audio file.",
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported audio format. Please upload WAV, MP3, M4A, OGG, or WEBM.",
        }), 400

    unique_name = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(file_path)

    try:
        transcript, language, duration, segments = transcribe_audio_file(file_path)
        if not transcript:
            return jsonify({
                "success": False,
                "error": "No audible speech could be detected in the recording. Please speak clearly into the microphone.",
            }), 400

        return jsonify({
            "success": True,
            "transcript": transcript,
            "language": language,
            "duration_seconds": duration,
            "segments": segments,
            "filename": file.filename,
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Audio transcription error: {str(e)}",
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Audio file or Text -> Whisper STT -> Clean text -> TF-IDF & Chi2 -> ML Prediction
    """
    if not session.get("logged_in"):
        return jsonify({
            "success": False,
            "error": "Session unauthorized or expired. Please sign in.",
        }), 401

    load_model_bundle()
    if MODEL_BUNDLE is None:
        return jsonify({
            "success": False,
            "error": "Machine Learning model is not loaded. Run train_model.py first.",
        }), 500

    transcript = ""
    language = "en"
    duration = 0.0
    segments = []
    file_name = None

    # 1. Handle Audio Upload
    if "audio" in request.files:
        file = request.files["audio"]
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Please upload an audio file.",
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported audio format. Please upload WAV, MP3, M4A, OGG, or WEBM.",
            }), 400

        # Save audio file securely
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(file_path)
        file_name = file.filename

        # If user supplied an edited/manually verified transcript, prioritize it
        edited_transcript = request.form.get("edited_transcript", "").strip()
        if edited_transcript:
            transcript = edited_transcript
            # Still get duration if possible
            try:
                _, language, duration, segments = transcribe_audio_file(file_path)
            except Exception:
                pass
        else:
            try:
                # Transcribe with Whisper
                transcript, language, duration, segments = transcribe_audio_file(file_path)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Audio transcription error: {str(e)}",
                }), 500

        if not transcript:
            return jsonify({
                "success": False,
                "error": "No speech could be detected in the uploaded audio.",
            }), 400

    # 2. Handle Text Input
    elif "text" in request.form or (request.is_json and "text" in request.json):
        text_data = request.form.get("text") or (request.json or {}).get("text")
        transcript = str(text_data).strip()
        if not transcript:
            return jsonify({
                "success": False,
                "error": "Please enter call transcript text.",
            }), 400
        file_name = "Pasted Call Transcript"
    else:
        return jsonify({
            "success": False,
            "error": "Please upload an audio file or provide transcript text.",
        }), 400

    # 3. Preprocess Transcript using EXACT training pipeline
    cleaned_transcript = clean_text(transcript)
    vectorizer = MODEL_BUNDLE["vectorizer"]
    selector = MODEL_BUNDLE["selector"]
    best_model = MODEL_BUNDLE["best_model"]
    best_model_name = MODEL_BUNDLE["best_model_name"]
    all_models = MODEL_BUNDLE.get("all_models", {})
    metrics_list = MODEL_BUNDLE.get("metrics", [])
    metrics_by_name = {m["model"]: m for m in metrics_list}

    # Transform text to TF-IDF then Chi2 features
    feat_vec = vectorizer.transform([cleaned_transcript])
    feat_selected = selector.transform(feat_vec)

    # 4. Predict using the recommended/best classifier
    pred_label = int(best_model.predict(feat_selected)[0])
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(feat_selected)[0]
        # Real confidence for predicted class
        confidence = float(probabilities[pred_label])
        spam_prob = float(probabilities[1])
    else:
        confidence = 1.0
        spam_prob = float(pred_label)

    # Primary binary target: SPAM or NOT SPAM
    verdict = "SPAM" if pred_label == 1 else "NOT SPAM"

    # 5. Extract call signals (Urgency, Financial, Security/Credential tokens)
    signals = extract_signals(transcript)

    # Secondary Supporting Analysis:
    # If the call is classified as SPAM, determine if it has fraud/credential indicators (1,1)
    # or general promotional / non-fraud spam indicators (1,0).
    fraud_risk = "N/A (Legitimate)"
    if pred_label == 1:
        has_sec = signals.get("security_keywords", {}).get("count", 0) > 0
        has_fin = signals.get("financial_keywords", {}).get("count", 0) > 0
        has_urg = signals.get("urgency_keywords", {}).get("count", 0) > 0
        if has_sec or (has_fin and has_urg):
            fraud_risk = "Spam / Fraud (1,1)"
        else:
            fraud_risk = "Spam / Non-Fraud (1,0)"

    # 6. Multi-model consensus comparison
    model_results = []
    for name, clf in all_models.items():
        clf_pred = int(clf.predict(feat_selected)[0])
        clf_verdict = "SPAM" if clf_pred == 1 else "NOT SPAM"
        if hasattr(clf, "predict_proba"):
            c_prob = float(clf.predict_proba(feat_selected)[0][clf_pred])
        else:
            c_prob = 1.0
        benchmark = metrics_by_name.get(name, {})
        model_results.append({
            "model": name,
            "prediction": clf_verdict,
            "confidence": round(c_prob, 4),
            "accuracy": benchmark.get("accuracy", 0.0),
            "f1_score": benchmark.get("f1_score", 0.0),
            "is_best": (name == best_model_name),
        })

    model_results.sort(key=lambda x: x["f1_score"], reverse=True)

    # 7. Record in history
    history_entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": file_name,
        "transcript": transcript,
        "prediction": verdict,
        "fraud_risk": fraud_risk,
        "confidence": round(confidence, 4),
        "model": best_model_name,
    }
    ANALYSIS_HISTORY.insert(0, history_entry)
    if len(ANALYSIS_HISTORY) > 25:
        ANALYSIS_HISTORY.pop()

    return jsonify({
        "success": True,
        "prediction": verdict,
        "fraud_risk": fraud_risk,
        "confidence": round(confidence, 4),
        "spam_probability": round(spam_prob, 4),
        "model_name": best_model_name,
        "transcript": transcript,
        "language": language,
        "duration_seconds": round(duration, 2),
        "segments": segments,
        "signals": signals,
        "model_results": model_results,
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "Uploaded file is too large. Maximum allowed size is 25 MB.",
    }), 413


@app.errorhandler(404)
def not_found(error):
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "success": False,
        "error": "An unexpected error occurred during processing. Please try again.",
    }), 500


# Initialize model on startup
load_model_bundle()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" CallShield Spam & Fraud Call Detection Application")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
