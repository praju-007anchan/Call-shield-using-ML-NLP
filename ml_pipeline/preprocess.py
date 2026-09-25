"""Shared transcript cleaning and engineered features for CallShield training."""
from __future__ import annotations

import re
from collections import Counter

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "has", "have",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "we", "with", "you", "your",
}
SIGNAL_GROUPS = {
    "urgency": ("urgent", "immediately", "expire", "final notice", "act now", "within 24 hours"),
    "money": ("refund", "prize", "cash", "payment", "bank", "loan", "credit", "reward"),
    "otp": ("otp", "one time password", "verification code", "security code", "pin"),
}


def clean_transcript(text: str) -> str:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9']+", lowered)
    return " ".join(token for token in tokens if token not in STOP_WORDS)


def engineered_features(text: str, duration_seconds: float = 0.0) -> dict[str, float]:
    lowered = text.lower()
    words = re.findall(r"\b[\w']+\b", lowered)
    features = {"word_count": float(len(words))}
    for group, keywords in SIGNAL_GROUPS.items():
        features[f"{group}_keywords"] = float(sum(lowered.count(keyword) for keyword in keywords))
    features["url_mentions"] = float(len(re.findall(r"https?://|www\.", lowered)))
    features["speech_rate"] = float(len(words) / (duration_seconds / 60)) if duration_seconds > 0 else 0.0
    return features


def top_terms(texts: list[str], limit: int = 12) -> list[tuple[str, int]]:
    return Counter(token for text in texts for token in clean_transcript(text).split()).most_common(limit)