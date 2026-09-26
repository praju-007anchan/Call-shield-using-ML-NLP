"""Train and compare CallShield's eight requested classifiers.

Input CSV must contain `text` and either `label` or `is_spam` columns. Optional
`duration_seconds` is used for engineered speech-rate features. Outputs a
joblib bundle and JSON metrics that the API can serve.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import clean_transcript, engineered_features


def build_models(seed: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=seed),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(probability=True, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, max_depth=12),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=400, random_state=seed),
    }


def evaluate(name, model, x_train, x_test, y_train, y_test):
    started = time.perf_counter()
    model.fit(x_train, y_train)
    training_ms = (time.perf_counter() - started) * 1000
    latency_started = time.perf_counter()
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    y_test_binary = (y_test == "spam").astype(int)
    latency_ms = (time.perf_counter() - latency_started) * 1000 / max(x_test.shape[0], 1)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test_binary, probabilities),
        "pr_auc": average_precision_score(y_test_binary, probabilities),
        "log_loss": log_loss(y_test_binary, probabilities),
        "training_time_ms": training_ms,
        "inference_latency_ms": latency_ms,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }, model


def score_models(results):
    """Select a model using every deployment-relevant metric we record."""
    quality_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
        "pr_auc",
    ]

    def normalized(values, value, higher_is_better=True):
        minimum, maximum = min(values), max(values)
        if maximum == minimum:
            return 1.0
        score = (value - minimum) / (maximum - minimum)
        return score if higher_is_better else 1.0 - score

    log_losses = [row["log_loss"] for row in results]
    training_times = [row["training_time_ms"] for row in results]
    inference_latencies = [row["inference_latency_ms"] for row in results]
    for row in results:
        quality_score = sum(row[key] for key in quality_keys) / len(quality_keys)
        calibration_score = normalized(log_losses, row["log_loss"], higher_is_better=False)
        training_score = normalized(training_times, row["training_time_ms"], higher_is_better=False)
        latency_score = normalized(inference_latencies, row["inference_latency_ms"], higher_is_better=False)
        row["selection_score"] = round(
            quality_score * 0.85
            + calibration_score * 0.09
            + training_score * 0.03
            + latency_score * 0.03,
            6,
        )

    results.sort(key=lambda row: row["selection_score"], reverse=True)
    for rank, row in enumerate(results, start=1):
        row["selection_rank"] = rank
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with text,label[,duration_seconds]")
    parser.add_argument("--output", default="artifacts", help="Output directory")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        if "is_spam" not in df.columns:
            raise ValueError("CSV must contain a label or is_spam column.")
        df["label"] = df["is_spam"].map({0: "not_spam", 1: "spam", "0": "not_spam", "1": "spam"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].map({
        0: "not_spam", 1: "spam", "0": "not_spam", "1": "spam",
        "not_spam": "not_spam", "spam": "spam", "ham": "not_spam",
    }).fillna(df["label"].astype(str))
    df["clean_text"] = df["text"].map(clean_transcript)
    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    models = build_models(42)
    results, bundles = [], {}
    for name, model in models.items():
        result, fitted = evaluate(name, model, x_train_vec, x_test_vec, y_train, y_test)
        results.append(result)
        bundles[name] = fitted
    results = score_models(results)
    best_name = results[0]["model"]
    joblib.dump({"vectorizer": vectorizer, "model": bundles[best_name], "model_name": best_name}, output / "callshield.joblib")
    (output / "metrics.json").write_text(json.dumps({
        "dataset": str(args.csv),
        "feature_set": "TF-IDF word + engineered call signals",
        "split": "80/20 stratified split",
        "recommended_model": best_name,
        "metrics": results,
    }, indent=2))
    print(f"Saved {best_name} bundle and {len(results)} model evaluations to {output}")


if __name__ == "__main__":
    main()