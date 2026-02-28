#!/usr/bin/env python3
"""Inspect the Speakyer database: show table counts and recent rows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speakyer.database import TABLES, get_connection, init
from speakyer.config import config


def truncate(value: object, width: int) -> str:
    s = str(value) if value is not None else "NULL"
    return s[:width] if len(s) <= width else s[: width - 1] + "…"


def main() -> None:
    if not config.db_path.exists():
        print(f"Database not found at {config.db_path} — initialising...")
        init()
        print("Done.\n")

    conn = get_connection()
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    print(f"Database: {config.db_path}\n")

    for table in TABLES:
        if table not in existing:
            print(f"  {table}: (table not found)\n")
            continue

        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        bar = "─" * 52
        print(bar)
        print(f"  {table}  ({count} row{'s' if count != 1 else ''})")
        print(bar)

        if count == 0:
            print("  (empty)\n")
            continue

        # Column names
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT 0")
        cols = [d[0] for d in cursor.description]
        col_w = min(16, max(len(c) for c in cols))
        widths = [max(len(c), col_w) for c in cols]

        header = "  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths))
        print(header)
        print("  " + "  ".join("-" * w for w in widths))

        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 3"
        ).fetchall()
        for row in reversed(rows):
            cells = [truncate(v, w).ljust(w) for v, w in zip(row, widths)]
            print("  " + "  ".join(cells))
        print()

    conn.close()


if __name__ == "__main__":
    main()
