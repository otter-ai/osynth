"""Build a voice cache (``voice_catalog.json`` + ``samples/``) from Common Voice.

Each synthetic persona is assigned one reference speaker from this cache; the
reference clip seeds a TTS voice-cloned enrollment (step 7/8). The input is a
Mozilla Common Voice *Spontaneous Speech* (SPS) release directory containing
``ss-corpus-<lang>.tsv`` and ``audios/``. One clip per ``client_id`` is kept
(the longest within the duration window), converted to 16 kHz mono WAV.

Usage::

    python scripts/prepare_voice_cache.py \
        --corpus-dir data/commonvoice/sps-corpus-3.0-en \
        --output-dir data/voice_cache --lang en

Requires ``ffmpeg`` on PATH. The corpus used for the released data was
additionally screened for clip quality; that screening is not included here.
Check the Common Voice terms before redistributing any derived audio.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

GENDER_MAP = {
    "male_masculine": "male",
    "female_feminine": "female",
    "transgender": "non-binary",
    "intersex": "non-binary",
    "non-binary": "non-binary",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--min-duration-ms", type=int, default=6000)
    parser.add_argument("--max-duration-ms", type=int, default=30000)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    tsv = args.corpus_dir / f"ss-corpus-{args.lang}.tsv"
    with tsv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            duration = int(row.get("duration_ms") or 0)
        except ValueError:
            continue
        text = (row.get("transcription") or "").strip()
        if not text or not args.min_duration_ms <= duration <= args.max_duration_ms:
            continue
        if args.lang == "en" and (row.get("split") or "").strip() not in {"train", "dev", "test"}:
            continue
        by_speaker[row["client_id"]].append(row)

    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    speakers: dict[str, dict] = {}
    for speaker_id, speaker_rows in sorted(by_speaker.items()):
        if args.max_speakers and len(speakers) >= args.max_speakers:
            break
        row = max(speaker_rows, key=lambda r: int(r["duration_ms"]))
        wav = samples_dir / f"{speaker_id}.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.corpus_dir / "audios" / row["audio_file"]),
             "-ar", "16000", "-ac", "1", str(wav)],
            capture_output=True,
        )
        if result.returncode != 0:
            continue
        speakers[speaker_id] = {
            "gender": GENDER_MAP.get((row.get("gender") or "").strip(), "unknown"),
            "audio_path": f"samples/{wav.name}",
            "text": row["transcription"].strip(),
            "language": args.lang,
            "duration_ms": int(row["duration_ms"]),
            "accent": (row.get("accents") or "").strip(),
            "source_audio": row["audio_file"],
        }

    catalog = args.output_dir / "voice_catalog.json"
    catalog.write_text(json.dumps({"speakers": speakers}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(speakers)} speakers to {catalog}")


if __name__ == "__main__":
    main()
