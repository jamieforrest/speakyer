#!/usr/bin/env python3
"""Pretty-print a transcript with timestamps for a given episode ID.

Usage:
    python scripts/show_transcript.py <episode_id>
    python scripts/show_transcript.py <episode_id> --words   # show word-level timestamps
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speakyer.config import config
from speakyer.database import get_connection, init
from speakyer.storage import LocalStorage


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show transcript for an episode.")
    parser.add_argument("episode_id", type=int, help="Episode ID from list_episodes.py")
    parser.add_argument(
        "--words", action="store_true", help="Show word-level timestamps from raw JSON"
    )
    args = parser.parse_args()

    if not config.db_path.exists():
        init()

    conn = get_connection()

    episode = conn.execute(
        "SELECT e.id, e.title, e.published_at, s.name AS source "
        "FROM episodes e JOIN sources s ON s.id = e.source_id WHERE e.id = ?",
        (args.episode_id,),
    ).fetchone()

    if not episode:
        print(f"Episode {args.episode_id} not found.")
        sys.exit(1)

    transcript = conn.execute(
        "SELECT id, raw_json_path, language_detected, duration_seconds "
        "FROM transcripts WHERE episode_id = ?",
        (args.episode_id,),
    ).fetchone()

    if not transcript:
        print(f"No transcript found for episode {args.episode_id}.")
        print("Run:  python scripts/run_pipeline.py --stage transcribe")
        sys.exit(1)

    dt = episode["published_at"]
    published = dt.strftime("%Y-%m-%d") if dt else "unknown"
    dur = transcript["duration_seconds"] or 0
    print(f"\nEpisode {episode['id']}: {episode['title']}")
    print(
        f"Source: {episode['source']}  |  Published: {published}  |  "
        f"Duration: {fmt_time(dur)}  |  Language: {transcript['language_detected']}"
    )
    print()

    if args.words and transcript["raw_json_path"]:
        # Load word-level detail from raw Whisper JSON.
        storage = LocalStorage(config.data_dir)
        try:
            raw = json.loads(storage.read(transcript["raw_json_path"]).decode())
            for seg in raw.get("segments", []):
                print(f"  [{fmt_time(seg['start'])}–{fmt_time(seg['end'])}] {seg['text'].strip()}")
                for w in seg.get("words", []):
                    print(f"      {fmt_time(w['start'])}  {w['word'].strip()}")
                print()
        except FileNotFoundError:
            print(f"Raw JSON not found at {transcript['raw_json_path']}")
    else:
        # Load segments from DB.
        segments = conn.execute(
            "SELECT text, start_time, end_time FROM transcript_segments "
            "WHERE transcript_id = ? ORDER BY segment_index",
            (transcript["id"],),
        ).fetchall()

        for seg in segments:
            ts = fmt_time(seg["start_time"])
            print(f"  [{ts}] {seg['text']}")

    print()
    conn.close()


if __name__ == "__main__":
    main()
