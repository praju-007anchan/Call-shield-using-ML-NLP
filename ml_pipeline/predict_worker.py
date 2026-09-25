"""Run the persisted CallShield classifier for one transcript from stdin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

from preprocess import clean_transcript


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("text is required")
    bundle = joblib.load(Path(args.bundle))
    probabilities = bundle["model"].predict_proba(bundle["vectorizer"].transform([clean_transcript(text)]))[0]
    classes = list(bundle["model"].classes_)
    spam_index = classes.index("spam") if "spam" in classes else 1
    probability = float(probabilities[spam_index])
    print(json.dumps({
        "label": "spam" if probability >= 0.5 else "not_spam",
        "probability": round(probability, 6),
        "model": bundle["model_name"],
    }))


if __name__ == "__main__":
    main()