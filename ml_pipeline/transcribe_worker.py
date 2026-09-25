"""Transcribe one audio file with Faster-Whisper and emit JSON to stdout."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from faster_whisper import WhisperModel


def confidence(segment) -> float:
    # Faster-Whisper exposes average log probability rather than a calibrated
    # confidence. Convert it into a useful bounded display value.
    return round(max(0.0, min(1.0, math.exp(segment.avg_logprob))), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium"])
    args = parser.parse_args()
    if not Path(args.audio).exists():
        raise FileNotFoundError(args.audio)

    whisper = WhisperModel(args.model, device="cpu", compute_type="int8")
    segments, info = whisper.transcribe(
        args.audio,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )
    segment_list = list(segments)
    transcript = " ".join(segment.text.strip() for segment in segment_list).strip()
    duration = max((float(segment.end) for segment in segment_list), default=0.0)
    print(json.dumps({
        "transcript": transcript,
        "language": info.language or "unknown",
        "durationSeconds": round(duration, 3),
        "segments": [
            {
                "id": index,
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text.strip(),
                "confidence": confidence(segment),
            }
            for index, segment in enumerate(segment_list)
        ],
    }))


if __name__ == "__main__":
    main()