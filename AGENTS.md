# AGENTS.md - Persistent Memory & Project Context

## Project Overview: CallShield
**Real-Time Spam and Fraud Call Detection with Instant User Alerts**

### Core Architecture & Workflow
1. **Audio Input**: User uploads call recording (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`), records live microphone, or pastes transcript.
2. **Audio Decoding**: Decoded directly in Python via `av` (PyAV) into 16kHz mono float32 numpy arrays. This completely avoids any external `ffmpeg.exe` binary dependency on Windows.
3. **Speech-to-Text**: Local open-source **OpenAI Whisper** model generates the call transcript, language, duration, and timestamped speech segments.
4. **Text Preprocessing**: Matches `spam_fraud_call_detection.ipynb` identically:
   - Lowercasing
   - Removal of bracketed placeholders (`r"\[.*?\]"`)
   - Removal of non-alphanumeric characters (`r"[^a-z\s]"`)
   - Whitespace normalization
   - Removal of NLTK English stopwords
   - WordNet lemmatization (`WordNetLemmatizer`)
5. **Feature Representation**:
   - TF-IDF Vectorizer (`ngram_range=(1,2)`, `max_features=3000`, `min_df=2`)
   - Chi-Square Feature Selection: `SelectKBest(chi2, k=1500)`
6. **Machine Learning Target & Models**:
   - Ground Truth: Derived binary target `spam_target` (`0` = Not Spam, `1` = Spam).
   - Label Combinations: `(0,0)` -> Not Spam (400 total), `(1,0)` -> Spam (221 calls), `(1,1)` -> Spam + Fraud (179 calls).
   - Target Leakage Prevention: `fraud` column is strictly excluded from model feature inputs.
   - Best / Deployed Algorithm: **Logistic Regression** (`C=10, max_iter=1000, random_state=42`)
   - Achieved Holdout Metrics: **98.75% Accuracy, 98.75% Precision, 98.75% Recall, 98.75% F1-Score, 0.9998 ROC-AUC**
   - 5 Candidate Models Evaluated: Logistic Regression, Support Vector Machine (SVC), Naive Bayes (MultinomialNB), Random Forest (RandomForestClassifier), K-Nearest Neighbors (KNN).
   - Serialized in `model.pkl` and `model_metrics.json`.
7. **Prediction Output**:
   - Strictly binary: `SPAM` or `NOT SPAM`
   - Real, calibrated prediction confidence from `predict_proba`
   - Secondary Supporting Risk Analysis: Spam / Non-Fraud (`1,0`) vs Spam / Fraud (`1,1`)
   - Extracted call signals: Urgency keywords, Financial keywords, Security/OTP keywords, Word count.
   - Multi-model consensus comparison table.

### Key Files & Structure
- `app.py`: Flask backend web server (runs on `http://127.0.0.1:5000`).
- `train_model.py`: Reproducible model training and serialization script matching the notebook.
- `spam_fraud_call_detection.ipynb`: Reference Colab notebook.
- `model.pkl`: Serialized model bundle (`vectorizer`, `selector`, `best_model`, `all_models`, `metrics`).
- `model_metrics.json`: JSON evaluation metrics for UI display.
- `users.json`: Persistent user account credentials store (survives server restarts).
- `dataset.csv`: Dataset with 800 phone call transcripts (`text`, `is_spam`, `is_confidential`).
- `templates/login.html`: Sign-in view with Quick Fill, Remember Me, and Password toggle.
- `templates/register.html`: User registration view saving accounts permanently to `users.json`.
- `templates/forgot_password.html`: Password reset view updating `users.json` permanently.
- `templates/index.html`: Main call analysis and live review workspace.
- `templates/models.html`: Model evaluation bench and confusion matrix.
- `static/css/style.css`: CallShield styling tokens (`Space Grotesk`, `DM Mono`, lime & spruce palette).
- `static/js/main.js`: Interactive audio handling, drag & drop, Whisper STT, prediction rendering, and session history.
- `static/samples/`: Preloaded sample audio files (`callshield_spam_call.mp3`, `callshield_not_spam_call.mp3`).
- `requirements.txt`: Python dependencies (`Flask`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `openai-whisper`, `av`, `nltk`).

### User Preferences & Guidelines
- Always preserve the clean Python/Flask structure.
- Do not add unnecessary Node/pnpm/Replit files back.
- When running locally, start with `python app.py` on `http://127.0.0.1:5000`.
- Persistent user accounts are stored in `users.json` (defaults: `admin` / `admin123`, `analyst` / `shield2026`).
