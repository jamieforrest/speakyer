#!/usr/bin/env python3
"""List fetched episodes with their source and processing status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speakyer.config import config
from speakyer.database import get_connection, init

STATUS_SYMBOLS = {
    "fetched": "○",
    "downloaded": "●",
    "transcribed": "◉",
    "processed": "◆",
    "exported": "✓",
}


def main() -> None:
    if not config.db_path.exists():
        init()

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT e.id, s.name AS source, e.title, e.published_at, e.status
        FROM episodes e
        JOIN sources s ON s.id = e.source_id
        ORDER BY e.published_at DESC
        LIMIT 50
        """
    ).fetchall()
    conn.close()

    if not rows:
        print("No episodes found. Run: python scripts/run_pipeline.py")
        return

    print(f"\n{'ID':<5}  {'S':<2}  {'Source':<20}  {'Published':<12}  Title")
    print("─" * 80)
    for row in rows:
        symbol = STATUS_SYMBOLS.get(row["status"], "?")
        published = (row["published_at"] or "")[:10]
        title = (row["title"] or "")[:48]
        print(f"{row['id']:<5}  {symbol}   {row['source']:<20}  {published:<12}  {title}")

    print()
    print("Status: ○ fetched  ● downloaded  ◉ transcribed  ◆ processed  ✓ exported")
    print()


if __name__ == "__main__":
    main()
