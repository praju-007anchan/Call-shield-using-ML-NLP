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
    vector = bundle["vectorizer"].transform([clean_transcript(text)])
    models = bundle.get("models") or {bundle["model_name"]: bundle["model"]}
    metrics_by_model = {
        row["model"]: row for row in bundle.get("metrics", [])
    }
    predictions = []
    for model_name, model in models.items():
        probabilities = model.predict_proba(vector)[0]
        classes = list(model.classes_)
        spam_index = classes.index("spam") if "spam" in classes else 1
        probability = float(probabilities[spam_index])
        benchmark = metrics_by_model.get(model_name, {})
        predictions.append({
            "model": model_name,
            "label": "spam" if probability >= 0.5 else "not_spam",
            "probability": round(probability, 6),
            "accuracy": round(float(benchmark.get("accuracy", 0)), 6),
            "precision": round(float(benchmark.get("precision_macro", 0)), 6),
            "recall": round(float(benchmark.get("recall_macro", 0)), 6),
            "f1": round(float(benchmark.get("f1_macro", 0)), 6),
            "selectionScore": round(float(benchmark.get("selection_score", 0)), 6),
            "selectionRank": int(benchmark.get("selection_rank", 0)),
        })
    predictions.sort(key=lambda row: row["selectionRank"] or 999)
    selected = predictions[0]
    print(json.dumps({
        "label": selected["label"],
        "probability": selected["probability"],
        "model": selected["model"],
        "modelResults": predictions,
    }))


if __name__ == "__main__":
    main()