#!/usr/bin/env python3
"""Preview Anki cards queued for an episode before export.

Usage:
    python scripts/show_cards.py <episode_id>
    python scripts/show_cards.py <episode_id> --level B1
    python scripts/show_cards.py <episode_id> --status pending
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speakyer.database import db, init

LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", None]


def _bold_word(sentence: str, surface_form: str) -> str:
    """Wrap the first case-insensitive occurrence of surface_form with **...**."""
    lower = sentence.lower()
    idx = lower.find(surface_form.lower())
    if idx == -1:
        return sentence
    return sentence[:idx] + "**" + sentence[idx : idx + len(surface_form)] + "**" + sentence[idx + len(surface_form):]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview cards for an episode.")
    parser.add_argument("episode_id", type=int, help="Episode DB id")
    parser.add_argument(
        "--level", choices=["A1", "A2", "B1", "B2", "C1", "C2"],
        help="Filter to a specific CEFR level",
    )
    parser.add_argument(
        "--status", choices=["pending", "exported"], default=None,
        help="Filter by card status (default: all)",
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
        SELECT c.id AS card_id, c.deck_name, c.status AS card_status,
               w.lemma, w.surface_form, w.pos, w.cefr_level,
               w.example_sentence, w.start_time,
               e.title AS episode_title, s.name AS source_name
        FROM cards c
        JOIN words w ON c.word_id = w.id
        JOIN episodes e ON w.episode_id = e.id
        JOIN sources s ON e.source_id = s.id
        WHERE w.episode_id = ?
    """
    params: list = [args.episode_id]
    if args.level:
        query += " AND w.cefr_level = ?"
        params.append(args.level)
    if args.status:
        query += " AND c.status = ?"
        params.append(args.status)
    query += " ORDER BY w.cefr_level, w.lemma"

    with db() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        msg = "No cards found."
        if ep["status"] not in ("cards_pending", "exported"):
            msg += " Run the cards stage first: python scripts/run_pipeline.py --stage cards"
        print(f"  {msg}")
        return

    print(f"  Cards: {len(rows)}\n")

    current_level = "UNSET"
    for row in rows:
        level = row["cefr_level"] or "untagged"
        if level != current_level:
            current_level = level
            print(f"  ── {level} ──────────────────────────────────────────")

        sentence = (row["example_sentence"] or "").strip()
        front = _bold_word(sentence, row["surface_form"])
        if len(front) > 100:
            front = front[:97] + "..."

        ts = f"{row['start_time']:.1f}s" if row["start_time"] is not None else "?"
        status_tag = f"[{row['card_status']}]" if row["card_status"] != "pending" else ""

        print(f"    [{row['card_id']}] {row['lemma']:<22} {row['pos']:<5} {level:<3}  @{ts}  {status_tag}")
        print(f"      Front: {front}")
        print(f"      Back:  {row['lemma']} · {row['pos']}"
              + (f" · CEFR {row['cefr_level']}" if row["cefr_level"] else "")
              + f" · {row['source_name']}")
        print()


if __name__ == "__main__":
    main()
