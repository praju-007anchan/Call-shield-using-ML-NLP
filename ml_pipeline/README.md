# CallShield ML pipeline

The web app uses Faster-Whisper for real audio-to-text transcription and a transparent heuristic classifier until a trained call model is available. This folder contains the production training path described in the build brief.

## Train

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ml_pipeline/requirements.txt
python ml_pipeline/train.py --csv data/calls.csv --output ml_pipeline/artifacts
```

The CSV requires `text` and `label` columns (`spam` / `not_spam` or `1` / `0`). Training writes `metrics.json` and `callshield.joblib`.

Datasets using `is_spam` instead of `label` are accepted directly. A supplied `is_confidential` column is preserved in the source data but is not used by the spam classifier.

## Run local audio inference

```bash
python ml_pipeline/inference.py recordings/incoming.mp3 \
  --bundle ml_pipeline/artifacts/callshield.joblib \
  --whisper-model base
```

`faster-whisper` accepts WAV, MP3, and M4A when ffmpeg is available. Whisper models are selectable with `tiny`, `base`, `small`, or `medium`; use `tiny` for fast iteration and `small`/`medium` for stronger transcription.

## Web app transcription

The web app sends uploaded WAV, MP3, M4A, and browser-recorded WebM files to `POST /api/transcribe`. The API temporarily stores the bytes, runs the selected Faster-Whisper model, returns the full transcript plus timestamped segments, and removes the temporary file.