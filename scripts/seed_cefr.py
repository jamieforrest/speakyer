#!/usr/bin/env python3
"""Seed the cefr_words table from a CSV file.

Usage:
    python scripts/seed_cefr.py                       # use bundled data/cefr/de.csv
    python scripts/seed_cefr.py path/to/words.csv     # use a custom CSV

CSV format (one header row, then data rows):
    lemma,level,pos

    lemma   — lowercase lemma as spaCy produces it (e.g. "gehen", "haus")
    level   — CEFR level: A1, A2, B1, B2, C1, or C2
    pos     — optional POS tag (VERB, NOUN, ADJ, ADV …); may be empty

Lines starting with '#' are treated as comments and skipped.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speakyer.database import db, init

DEFAULT_CSV = Path(__file__).parent.parent / "data" / "cefr" / "de.csv"

VALID_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


def seed(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, str, str | None]] = []
    skipped = 0

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(
            (line for line in fh if not line.lstrip().startswith("#"))
        )
        for row in reader:
            lemma = row.get("lemma", "").strip().lower()
            level = row.get("level", "").strip().upper()
            pos = row.get("pos", "").strip() or None

            if not lemma or level not in VALID_LEVELS:
                skipped += 1
                continue

            rows.append((lemma, level, pos))

    if not rows:
        print("No valid rows found in CSV.")
        return

    init()

    with db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO cefr_words (lemma, level, pos) VALUES (?, ?, ?)",
            rows,
        )

    print(f"Seeded {len(rows)} CEFR words (skipped {skipped} invalid rows).")

    # Show a summary by level.
    with db() as conn:
        counts = conn.execute(
            "SELECT level, COUNT(*) FROM cefr_words GROUP BY level ORDER BY level"
        ).fetchall()
    for level, count in counts:
        print(f"  {level}: {count}")


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    seed(csv_path)
