#!/usr/bin/env python3
"""Display vocabulary extracted for an episode, grouped by CEFR level.

Usage:
    python scripts/show_words.py <episode_id>
    python scripts/show_words.py <episode_id> --level B1
    python scripts/show_words.py <episode_id> --unique
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speakyer.database import db, init

LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show vocabulary for an episode.")
    parser.add_argument("episode_id", type=int, help="Episode DB id")
    parser.add_argument(
        "--level",
        choices=["A1", "A2", "B1", "B2", "C1", "C2"],
        help="Filter to a specific CEFR level",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Show each lemma only once (first occurrence)",
    )
    args = parser.parse_args()

    init()

    with db() as conn:
        ep = conn.execute(
            "SELECT title, status FROM episodes WHERE id = ?", (args.episode_id,)
        ).fetchone()

    if not ep:
        print(f"Episode {args.episode_id} not found.")
        sys.exit(1)

    print(f"\nEpisode {args.episode_id}: {ep['title']!r}  [{ep['status']}]")

    query = """
        SELECT w.lemma, w.surface_form, w.pos, w.cefr_level, w.example_sentence,
               w.start_time
        FROM words w
        WHERE w.episode_id = ?
    """
    params: list = [args.episode_id]
    if args.level:
        query += " AND w.cefr_level = ?"
        params.append(args.level)
    query += " ORDER BY w.start_time, w.id"

    with db() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        print("  No words found. Run the NLP stage first.")
        return

    # Group by CEFR level.
    by_level: dict[str | None, list] = defaultdict(list)
    seen_lemmas: set[str] = set()

    for row in rows:
        if args.unique:
            if row["lemma"] in seen_lemmas:
                continue
            seen_lemmas.add(row["lemma"])
        by_level[row["cefr_level"]].append(row)

    total = sum(len(v) for v in by_level.values())
    print(f"  Total words: {total}\n")

    for level in LEVEL_ORDER:
        words = by_level.get(level)
        if not words:
            continue
        label = level if level else "untagged"
        print(f"  ── {label} ({len(words)}) ──────────────────────────────")
        for row in words:
            ts = f"{row['start_time']:.1f}s" if row["start_time"] is not None else "?"
            sentence = (row["example_sentence"] or "").strip()
            if len(sentence) > 80:
                sentence = sentence[:77] + "..."
            print(f"    {row['surface_form']:<20} [{row['lemma']:<20}] {row['pos']:<5}  @{ts}")
            if sentence:
                print(f"      {sentence}")
        print()


if __name__ == "__main__":
    main()
