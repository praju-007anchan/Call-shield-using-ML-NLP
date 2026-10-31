"""
CallShield - Real-Time Spam and Fraud Call Detection
Model Training & Evaluation Pipeline

Workflow:
1. Raw Dataset Loading
2. Data Cleaning & Validation
3. Label Validation:
   - (0,0): Legitimate / Not Spam
   - (1,0): Spam, but not necessarily Fraud
   - (1,1): Spam + Fraud
   (Do NOT delete, merge, or change 1,0 to 0,0)
4. Derived Target Creation:
   - if spam == 0 -> spam_target = 0 (Not Spam)
   - if spam == 1 -> spam_target = 1 (Spam)
   Therefore:
     0,0 -> Not Spam
     1,0 -> Spam
     1,1 -> Spam
5. Primary EDA Statistics (Spam vs Not Spam):
   - Spam count = count(1,0) + count(1,1)
   - Not Spam count = count(0,0)
6. Secondary Supporting Analysis (Spam Composition):
   - Among Spam: Spam / Non-Fraud (1,0) vs Spam / Fraud (1,1)
7. Text Preprocessing: lowercasing, bracket removal, non-alphanumeric removal, lemmatization, stop words
8. Train/Test Split: 80/20 Stratified on spam_target
9. Feature Extraction: TF-IDF (1-2 ngrams, min_df=2) + Chi2 SelectKBest (k=1500)
   (fraud column strictly excluded from features to prevent target leakage)
10. Train 5 Models:
    - Logistic Regression
    - Naive Bayes (MultinomialNB)
    - Support Vector Machine (SVC)
    - K-Nearest Neighbors (KNN)
    - Random Forest (RandomForestClassifier)
11. Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, 2x2 Binary Confusion Matrix
12. Hyperparameter Tuning: GridSearchCV for best model
13. Model Export: model.pkl and model_metrics.json
"""

import json
import os
import re
import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Ensure NLTK resources without blocking network lookups
for _res, _pkg in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Preprocess text exactly as defined in the training pipeline."""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    )


def extract_signal_counts(text: str) -> dict:
    """Extract contextual call risk signals for UI display."""
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


def train_and_export():
    dataset_candidates = [
        "dataset.csv",
        "attached_assets/dataset_1790343886829.csv",
        "attached_assets/dataset.csv",
    ]
    dataset_path = None
    for candidate in dataset_candidates:
        if os.path.exists(candidate):
            dataset_path = candidate
            break

    if not dataset_path:
        raise FileNotFoundError("Could not locate dataset CSV file.")

    print("=" * 60)
    print("CallShield ML Training & Evaluation Pipeline")
    print("=" * 60)
    print(f"[1] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    total_raw_rows = len(df)
    print(f"    Raw dataset shape: {df.shape}")

    # Determine column names
    spam_col = "spam" if "spam" in df.columns else "is_spam"
    fraud_col = "fraud" if "fraud" in df.columns else "is_confidential"

    if spam_col not in df.columns or fraud_col not in df.columns:
        raise ValueError(f"Dataset missing required spam or fraud columns. Found: {list(df.columns)}")

    print(f"    Detected spam column: '{spam_col}', fraud column: '{fraud_col}'")

    # =========================================================================
    # [2] Label Validation
    # Correct meaning of combinations:
    # 0,0 -> Legitimate / Not Spam
    # 1,0 -> Spam, but not necessarily Fraud
    # 1,1 -> Spam + Fraud
    # =========================================================================
    print("\n[2] Validating Label Combinations:")
    c_0_0 = int(((df[spam_col] == 0) & (df[fraud_col] == 0)).sum())
    c_1_0 = int(((df[spam_col] == 1) & (df[fraud_col] == 0)).sum())
    c_1_1 = int(((df[spam_col] == 1) & (df[fraud_col] == 1)).sum())
    c_0_1 = int(((df[spam_col] == 0) & (df[fraud_col] == 1)).sum())

    print(f"    • (0,0) Legitimate / Not Spam:           {c_0_0} calls")
    print(f"    • (1,0) Spam, but not necessarily Fraud: {c_1_0} calls (KEPT as Spam)")
    print(f"    • (1,1) Spam + Fraud:                    {c_1_1} calls (KEPT as Spam)")
    if c_0_1 > 0:
        print(f"    • (0,1) Legitimate (confidential flag):  {c_0_1} calls")

    # =========================================================================
    # [3] Create Derived Target: spam_target
    # Rule:
    #   if spam == 0: spam_target = 0 (Not Spam)
    #   if spam == 1: spam_target = 1 (Spam)
    # Therefore:
    #   0,0 -> Not Spam
    #   1,0 -> Spam
    #   1,1 -> Spam
    # =========================================================================
    print("\n[3] Creating Derived Target 'spam_target':")
    df["spam_target"] = df[spam_col].apply(lambda s: 1 if s == 1 else 0)

    not_spam_count = int((df["spam_target"] == 0).sum())
    spam_count = int((df["spam_target"] == 1).sum())

    print(f"    • Primary Binary Target Counts:")
    print(f"      - Not Spam (0): {not_spam_count} calls (0,0 + legitimate)")
    print(f"      - Spam (1):     {spam_count} calls (sum of 1,0 [{c_1_0}] + 1,1 [{c_1_1}])")

    # =========================================================================
    # [4] Secondary Analysis: Spam Composition (Among Spam Records)
    # Supporting analysis only. Does NOT alter the primary binary classification.
    # =========================================================================
    spam_non_fraud_pct = round((c_1_0 / spam_count) * 100, 2) if spam_count > 0 else 0.0
    spam_fraud_pct = round((c_1_1 / spam_count) * 100, 2) if spam_count > 0 else 0.0

    print("\n[4] Secondary Supporting Analysis (Spam Composition):")
    print(f"    Among {spam_count} total Spam records:")
    print(f"    • Spam / Non-Fraud (1,0): {c_1_0} ({spam_non_fraud_pct}%)")
    print(f"    • Spam / Fraud (1,1):     {c_1_1} ({spam_fraud_pct}%)")

    # =========================================================================
    # [5] Text Preprocessing
    # =========================================================================
    print("\n[5] Preprocessing Call Transcripts...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Ensure no empty cleaned strings
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"    Cleaned records ready for vectorization: {len(df)}")

    # =========================================================================
    # [6] Feature Extraction (TF-IDF + Chi2 Feature Selection)
    # TARGET LEAKAGE PREVENTION:
    # The 'fraud' column is strictly NOT used as an input feature because it is
    # directly linked to the call labels. Features are generated purely from text!
    # =========================================================================
    print("\n[6] Feature Extraction (Target Leakage Prevention Active):")
    print("    • Vectorizing call text via TF-IDF (1-2 ngrams, min_df=2, max_features=3000)...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=2)
    X_full = tfidf.fit_transform(df["clean_text"])
    y = df["spam_target"].values

    k = min(1500, X_full.shape[1])
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X_full, y)
    print(f"    • Chi-Square SelectKBest selected top k={k} text features.")
    print(f"    • Final feature matrix shape: {X_selected.shape}")

    # =========================================================================
    # [7] 80/20 Stratified Train / Test Split on spam_target
    # =========================================================================
    print("\n[7] Splitting Data (80% Train / 20% Test, Stratified on spam_target)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    • Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # =========================================================================
    # [8] Train 5 Candidate ML Models
    # 1. Logistic Regression
    # 2. Naive Bayes (MultinomialNB)
    # 3. Support Vector Machine (SVC)
    # 4. K-Nearest Neighbors (KNN)
    # 5. Random Forest (RandomForestClassifier)
    # =========================================================================
    print("\n[8] Training & Evaluating 5 Candidate Binary Classifiers:")
    models = {
        "Logistic Regression": LogisticRegression(
            C=10, max_iter=1000, random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Support Vector Machine": SVC(
            C=1.0, kernel="linear", probability=True, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
    }

    metrics = []
    trained_models = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        else:
            roc_auc = 0.0

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        trained_models[name] = clf
        metrics.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
        })
        print(f"    • [{name:24s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    # =========================================================================
    # [9] Hyperparameter Tuning for Best Performing Classifier
    # =========================================================================
    print("\n[9] Hyperparameter Tuning (GridSearchCV on Logistic Regression):")
    param_grid = {
        "C": [0.1, 1, 5, 10, 20],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_tuned_lr = grid.best_estimator_
    y_pred_tuned = best_tuned_lr.predict(X_test)
    tuned_f1 = float(f1_score(y_test, y_pred_tuned))
    tuned_acc = float(accuracy_score(y_test, y_pred_tuned))
    print(f"    • Best Parameters: {grid.best_params_}")
    print(f"    • Best 5-Fold CV F1: {grid.best_score_:.4f}")
    print(f"    • Tuned Model Test Accuracy: {tuned_acc:.4f} | Test F1: {tuned_f1:.4f}")

    # Rank candidate models by F1-Score
    metrics.sort(key=lambda m: m["f1_score"], reverse=True)
    for rank, m in enumerate(metrics, start=1):
        m["rank"] = rank

    best_model_name = metrics[0]["model"]
    best_clf = trained_models[best_model_name]
    print(f"\n[10] Selected Best Model for Deployment: {best_model_name}")

    # =========================================================================
    # [11] Serialize Pipeline & Metrics
    # =========================================================================
    bundle = {
        "vectorizer": tfidf,
        "selector": selector,
        "best_model": best_clf,
        "best_model_name": best_model_name,
        "all_models": trained_models,
        "metrics": metrics,
        "classes": [0, 1],
        "class_labels": {0: "NOT SPAM", 1: "SPAM"},
        "tuning_info": {
            "best_params": grid.best_params_,
            "best_cv_f1": round(float(grid.best_score_), 4),
            "test_accuracy": round(tuned_acc, 4),
            "test_f1": round(tuned_f1, 4),
        },
    }

    joblib.dump(bundle, "model.pkl")
    print("    • Saved model bundle -> model.pkl")

    metrics_data = {
        "dataset_name": os.path.basename(dataset_path),
        "total_records": len(df),
        "target_variable": "spam_target",
        "labels_meaning": {
            "0,0": "Legitimate / Not Spam",
            "1,0": "Spam, but not necessarily Fraud",
            "1,1": "Spam + Fraud",
        },
        "eda_summary": {
            "not_spam_count": not_spam_count,
            "spam_count": spam_count,
            "combination_counts": {
                "0,0": c_0_0,
                "1,0": c_1_0,
                "1,1": c_1_1,
                "0,1": c_0_1,
            },
            "spam_composition": {
                "non_fraud_count": c_1_0,
                "fraud_count": c_1_1,
                "non_fraud_pct": spam_non_fraud_pct,
                "fraud_pct": spam_fraud_pct,
            },
        },
        "target_leakage_note": "The 'fraud' column is strictly excluded from input features to avoid target leakage. All models rely solely on NLP signals from the spoken call transcript.",
        "split": "80% Train / 20% Test (Stratified on spam_target)",
        "features": f"TF-IDF (1-2 ngrams, min_df=2) + Chi2 SelectKBest (k={k})",
        "recommended_model": best_model_name,
        "tuning_info": {
            "best_params": grid.best_params_,
            "best_cv_f1": round(float(grid.best_score_), 4),
            "test_accuracy": round(tuned_acc, 4),
            "test_f1": round(tuned_f1, 4),
        },
        "metrics": metrics,
    }

    with open("model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print("    • Saved metrics & EDA -> model_metrics.json")
    print("\nTraining and export completed successfully!")


if __name__ == "__main__":
    train_and_export()
