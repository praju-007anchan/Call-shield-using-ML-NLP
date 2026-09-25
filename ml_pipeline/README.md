# CallShield ML pipeline

The web app ships with transparent demo metrics and a heuristic transcript classifier so it is usable before model weights and a labeled dataset are available. This folder contains the production training path described in the build brief.

## Train

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ml_pipeline/requirements.txt
python ml_pipeline/train.py --csv data/calls.csv --output ml_pipeline/artifacts
```

The CSV requires `text` and `label` columns (`spam` / `not_spam` or `1` / `0`). Training writes `metrics.json` and `callshield.joblib`.

## Run local audio inference

```bash
python ml_pipeline/inference.py recordings/incoming.mp3 \
  --bundle ml_pipeline/artifacts/callshield.joblib \
  --whisper-model base
```

`faster-whisper` accepts WAV, MP3, and M4A when ffmpeg is available. Whisper models are selectable with `tiny`, `base`, `small`, or `medium`; use `tiny` for fast iteration and `small`/`medium` for stronger transcription.