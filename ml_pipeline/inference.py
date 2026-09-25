"""Local audio -> Whisper transcript -> persisted classifier inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from faster_whisper import WhisperModel

from preprocess import clean_transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--bundle", default="ml_pipeline/artifacts/callshield.joblib")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium"])
    args = parser.parse_args()
    whisper = WhisperModel(args.whisper_model, compute_type="int8")
    segments, info = whisper.transcribe(args.audio, vad_filter=True)
    segments = list(segments)
    transcript = " ".join(segment.text.strip() for segment in segments)
    bundle = joblib.load(Path(args.bundle))
    prediction = bundle["model"].predict_proba(bundle["vectorizer"].transform([clean_transcript(transcript)]))[0, 1]
    print(json.dumps({
        "transcript": transcript,
        "language": info.language,
        "model": bundle["model_name"],
        "label": "spam" if prediction >= 0.5 else "not_spam",
        "probability": round(float(prediction), 4),
        "segments": [{"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments],
    }, indent=2))


if __name__ == "__main__":
    main()