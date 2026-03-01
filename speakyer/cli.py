import sys

import click

from speakyer import __version__


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Speakyer: podcast vocabulary learning pipeline."""


# ---------------------------------------------------------------------------
# Pipeline stages shared between `run` and `audio-update`
# ---------------------------------------------------------------------------

STAGES = {
    "download": "DownloadStage",
    "transcribe": "TranscribeStage",
    "nlp": "NLPStage",
    "cards": "CardGenStage",
    "audio-clip": "AudioClipStage",
    "export": "ExportStage",
}

STAGE_PENDING_STATUSES = {
    "download": {"fetched"},
    "transcribe": {"fetched", "downloaded"},
    "nlp": {"fetched", "downloaded", "transcribed"},
    "cards": {"fetched", "downloaded", "transcribed", "analyzed"},
    "audio-clip": {"fetched", "downloaded", "transcribed", "analyzed", "cards_pending"},
    "export": {"fetched", "downloaded", "transcribed", "analyzed", "cards_pending"},
}


def _load_stages(names: list[str]) -> list:
    from speakyer.pipeline import stages as mod

    return [getattr(mod, STAGES[n])() for n in names]


# ---------------------------------------------------------------------------
# speakyer run
# ---------------------------------------------------------------------------


@main.command()
@click.option("--source", default=None, help="Source name to process (default: all active)")
@click.option("--dry-run", is_flag=True, help="Show what would be done without writing")
@click.option("--episode-id", type=int, default=None, help="Process a specific episode by ID")
@click.option(
    "--stage",
    type=click.Choice(list(STAGES)),
    default=None,
    help="Run up to and including this stage (default: all)",
)
@click.option("--whisper-model", default=None, metavar="MODEL", help="Override the Whisper model for this run")
def run(source: str | None, dry_run: bool, episode_id: int | None, stage: str | None, whisper_model: str | None) -> None:
    """Fetch, transcribe, and export vocabulary from podcasts."""
    import speakyer.config as _cfg_module
    from speakyer.database import init
    from speakyer.pipeline.base import PipelineContext
    from speakyer.sources.loader import (
        get_episode_by_id,
        get_known_guids,
        get_pending_episodes,
        insert_episodes,
        sync_sources,
    )
    from speakyer.sources.rss import RSSPodcastSource

    if whisper_model:
        _cfg_module.config.whisper_model = whisper_model
        click.echo(f"Using Whisper model: {whisper_model}")

    init()

    stage_names = list(STAGES)
    if stage:
        stage_names = stage_names[: stage_names.index(stage) + 1]
    stages = _load_stages(stage_names)

    pending_statuses: set[str] = set()
    for n in stage_names:
        pending_statuses |= STAGE_PENDING_STATUSES[n]

    # --- Single episode mode ---
    if episode_id:
        result = get_episode_by_id(episode_id)
        if not result:
            click.echo(f"Episode {episode_id} not found.")
            sys.exit(1)
        ep, src = result
        click.echo(f"\n[{src.name}] Processing episode {ep.id}: {ep.title!r}")
        ctx = PipelineContext(source_config=src, episode=ep, dry_run=dry_run)
        for s in stages:
            ctx = s.run(ctx)
        downloaded = 1 if ctx.audio_path else 0
        click.echo(f"\nDone — 1 episode, {downloaded} downloaded.")
        return

    # --- Source-based mode ---
    sources = sync_sources()
    if not sources:
        click.echo("No active sources found. Add entries to sources.yaml.")
        sys.exit(1)

    if source:
        sources = [s for s in sources if s.name == source]
        if not sources:
            click.echo(f"Source {source!r} not found or not active.")
            sys.exit(1)

    rss = RSSPodcastSource()
    total_new = total_downloaded = 0

    for src in sources:
        click.echo(f"\n[{src.name}] Checking for new episodes...")
        known = get_known_guids(src.id)
        new_episodes = rss.get_new_episodes(src, known)

        if dry_run:
            total_new += len(new_episodes)
            episodes_to_process = new_episodes
        else:
            inserted = insert_episodes(new_episodes)
            total_new += len(inserted)
            episodes_to_process = get_pending_episodes(src.id, pending_statuses)

        if not episodes_to_process:
            click.echo("  No episodes to process.")
            continue

        click.echo(f"  {len(episodes_to_process)} episode(s) to process.")

        for ep in episodes_to_process:
            ctx = PipelineContext(source_config=src, episode=ep, dry_run=dry_run)
            for s in stages:
                ctx = s.run(ctx)
            if ctx.audio_path:
                total_downloaded += 1

    click.echo(f"\nDone — {total_new} new episode(s), {total_downloaded} downloaded.")


# ---------------------------------------------------------------------------
# speakyer audio-update
# ---------------------------------------------------------------------------


@main.command("audio-update")
@click.option("--source", default=None, help="Source name (default: all)")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
@click.option("--episode-id", type=int, default=None, help="Single episode by ID")
@click.option("--resync-tags", is_flag=True, help="Re-push fields/tags to all exported cards")
def audio_update(source: str | None, dry_run: bool, episode_id: int | None, resync_tags: bool) -> None:
    """Add audio clips to existing Anki cards and sync fields/tags."""
    from speakyer.database import db, init
    from speakyer.pipeline.base import PipelineContext
    from speakyer.pipeline.stages import AUDIO_UPDATE_PIPELINE
    from speakyer.sources.base import Episode, SourceConfig

    init()

    status_filter = "('pending', 'exported', 'audio_updated')" if resync_tags else "('pending', 'exported')"

    with db() as conn:
        ep_rows = conn.execute(
            f"""
            SELECT DISTINCT e.id, e.guid, e.title, e.audio_path, e.status,
                            e.source_id,
                            s.name AS source_name, s.rss_url,
                            s.language, s.transcript_strategy, s.active
            FROM episodes e
            JOIN words w ON w.episode_id = e.id
            JOIN cards c ON c.word_id = w.id
            JOIN sources s ON s.id = e.source_id
            WHERE c.status IN {status_filter}
            ORDER BY e.id
            """,
        ).fetchall()

    if episode_id:
        ep_rows = [r for r in ep_rows if r["id"] == episode_id]
    if source:
        ep_rows = [r for r in ep_rows if r["source_name"] == source]

    if not ep_rows:
        click.echo("No episodes with cards found.")
        return

    stages = [cls() for cls in AUDIO_UPDATE_PIPELINE]
    click.echo(f"Running audio-update pipeline for {len(ep_rows)} episode(s)...")

    for row in ep_rows:
        ep = Episode(
            id=row["id"],
            guid=row["guid"],
            source_id=row["source_id"],
            title=row["title"],
            audio_path=row["audio_path"],
            status=row["status"],
        )
        src = SourceConfig(
            id=row["source_id"],
            name=row["source_name"],
            rss_url=row["rss_url"],
            language=row["language"],
            transcript_strategy=row["transcript_strategy"],
            active=bool(row["active"]),
        )
        click.echo(f"\n[{src.name}] {ep.title!r}")
        ctx = PipelineContext(
            source_config=src, episode=ep,
            dry_run=dry_run, resync_tags=resync_tags,
        )
        for s in stages:
            ctx = s.run(ctx)


# ---------------------------------------------------------------------------
# speakyer fix-clips
# ---------------------------------------------------------------------------


@main.command("fix-clips")
def fix_clips() -> None:
    """Remove audio clips generated from Whisper-hallucinated segments."""
    from speakyer.nlp.audio_clipper import purge_hallucinated_clips

    purged = purge_hallucinated_clips()
    if not purged:
        click.echo("No bad clips found — all cards look good.")
        return
    for item in purged:
        click.echo(f"  purged card {item['card_id']}: {item['clip_path']}")
    click.echo(f"Removed {len(purged)} bad clip(s).")


# ---------------------------------------------------------------------------
# speakyer episodes
# ---------------------------------------------------------------------------


@main.command()
def episodes() -> None:
    """List fetched episodes with source and processing status."""
    from speakyer.config import config
    from speakyer.database import get_connection, init

    STATUS_SYMBOLS = {
        "fetched": "○",
        "downloaded": "●",
        "transcribed": "◉",
        "analyzed": "◆",
        "cards_pending": "▲",
        "exported": "✓",
    }

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
        """,
    ).fetchall()
    conn.close()

    if not rows:
        click.echo("No episodes found. Run: speakyer run")
        return

    click.echo(f"\n{'ID':<5}  {'S':<2}  {'Source':<20}  {'Published':<12}  Title")
    click.echo("─" * 80)
    for row in rows:
        symbol = STATUS_SYMBOLS.get(row["status"], "?")
        dt = row["published_at"]
        published = dt.strftime("%Y-%m-%d") if dt else ""
        title = (row["title"] or "")[:48]
        click.echo(f"{row['id']:<5}  {symbol}   {row['source']:<20}  {published:<12}  {title}")

    click.echo()
    click.echo("Status: ○ fetched  ● downloaded  ◉ transcribed  ◆ analyzed  ▲ cards_pending  ✓ exported")
    click.echo()


# ---------------------------------------------------------------------------
# speakyer transcript
# ---------------------------------------------------------------------------


@main.command()
@click.argument("episode_id", type=int)
@click.option("--words", is_flag=True, help="Show word-level timestamps from raw JSON")
def transcript(episode_id: int, words: bool) -> None:
    """Pretty-print a transcript for EPISODE_ID."""
    import json

    from speakyer.config import config
    from speakyer.database import get_connection, init
    from speakyer.storage import LocalStorage

    def fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    if not config.db_path.exists():
        init()

    conn = get_connection()

    episode = conn.execute(
        "SELECT e.id, e.title, e.published_at, s.name AS source "
        "FROM episodes e JOIN sources s ON s.id = e.source_id WHERE e.id = ?",
        (episode_id,),
    ).fetchone()

    if not episode:
        click.echo(f"Episode {episode_id} not found.")
        sys.exit(1)

    tr = conn.execute(
        "SELECT id, raw_json_path, language_detected, duration_seconds "
        "FROM transcripts WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()

    if not tr:
        click.echo(f"No transcript found for episode {episode_id}.")
        click.echo("Run:  speakyer run --stage transcribe")
        sys.exit(1)

    dt = episode["published_at"]
    published = dt.strftime("%Y-%m-%d") if dt else "unknown"
    dur = tr["duration_seconds"] or 0
    click.echo(f"\nEpisode {episode['id']}: {episode['title']}")
    click.echo(
        f"Source: {episode['source']}  |  Published: {published}  |  "
        f"Duration: {fmt_time(dur)}  |  Language: {tr['language_detected']}"
    )
    click.echo()

    if words and tr["raw_json_path"]:
        storage = LocalStorage(config.data_dir)
        try:
            raw = json.loads(storage.read(tr["raw_json_path"]).decode())
            for seg in raw.get("segments", []):
                click.echo(f"  [{fmt_time(seg['start'])}–{fmt_time(seg['end'])}] {seg['text'].strip()}")
                for w in seg.get("words", []):
                    click.echo(f"      {fmt_time(w['start'])}  {w['word'].strip()}")
                click.echo()
        except FileNotFoundError:
            click.echo(f"Raw JSON not found at {tr['raw_json_path']}")
    else:
        segments = conn.execute(
            "SELECT text, start_time, end_time FROM transcript_segments "
            "WHERE transcript_id = ? ORDER BY segment_index",
            (tr["id"],),
        ).fetchall()
        for seg in segments:
            ts = fmt_time(seg["start_time"])
            click.echo(f"  [{ts}] {seg['text']}")

    click.echo()
    conn.close()


# ---------------------------------------------------------------------------
# speakyer words
# ---------------------------------------------------------------------------


@main.command()
@click.argument("episode_id", type=int)
@click.option("--level", type=click.Choice(["A1", "A2", "B1", "B2", "C1", "C2"]), default=None, help="Filter to a CEFR level")
@click.option("--unique", is_flag=True, help="Show each lemma only once")
def words(episode_id: int, level: str | None, unique: bool) -> None:
    """Display vocabulary extracted for EPISODE_ID, grouped by CEFR level."""
    from collections import defaultdict

    from speakyer.database import db, init

    LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", None]

    init()

    with db() as conn:
        ep = conn.execute(
            "SELECT title, status FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()

    if not ep:
        click.echo(f"Episode {episode_id} not found.")
        sys.exit(1)

    click.echo(f"\nEpisode {episode_id}: {ep['title']!r}  [{ep['status']}]")

    query = """
        SELECT w.lemma, w.surface_form, w.pos, w.cefr_level, w.example_sentence,
               w.start_time
        FROM words w
        WHERE w.episode_id = ?
    """
    params: list = [episode_id]
    if level:
        query += " AND w.cefr_level = ?"
        params.append(level)
    query += " ORDER BY w.start_time, w.id"

    with db() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        click.echo("  No words found. Run the NLP stage first.")
        return

    by_level: dict[str | None, list] = defaultdict(list)
    seen_lemmas: set[str] = set()

    for row in rows:
        if unique:
            if row["lemma"] in seen_lemmas:
                continue
            seen_lemmas.add(row["lemma"])
        by_level[row["cefr_level"]].append(row)

    total = sum(len(v) for v in by_level.values())
    click.echo(f"  Total words: {total}\n")

    for lev in LEVEL_ORDER:
        wds = by_level.get(lev)
        if not wds:
            continue
        label = lev if lev else "untagged"
        click.echo(f"  ── {label} ({len(wds)}) ──────────────────────────────")
        for row in wds:
            ts = f"{row['start_time']:.1f}s" if row["start_time"] is not None else "?"
            sentence = (row["example_sentence"] or "").strip()
            if len(sentence) > 80:
                sentence = sentence[:77] + "..."
            click.echo(f"    {row['surface_form']:<20} [{row['lemma']:<20}] {row['pos']:<5}  @{ts}")
            if sentence:
                click.echo(f"      {sentence}")
        click.echo()


# ---------------------------------------------------------------------------
# speakyer cards
# ---------------------------------------------------------------------------


@main.command()
@click.argument("episode_id", type=int)
@click.option("--level", type=click.Choice(["A1", "A2", "B1", "B2", "C1", "C2"]), default=None, help="Filter to a CEFR level")
@click.option("--status", type=click.Choice(["pending", "exported"]), default=None, help="Filter by card status")
def cards(episode_id: int, level: str | None, status: str | None) -> None:
    """Preview Anki cards queued for EPISODE_ID."""
    from speakyer.database import db, init

    def _bold_word(sentence: str, surface_form: str) -> str:
        lower = sentence.lower()
        idx = lower.find(surface_form.lower())
        if idx == -1:
            return sentence
        return sentence[:idx] + "**" + sentence[idx: idx + len(surface_form)] + "**" + sentence[idx + len(surface_form):]

    init()

    with db() as conn:
        ep = conn.execute(
            "SELECT title, status FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()

    if not ep:
        click.echo(f"Episode {episode_id} not found.")
        sys.exit(1)

    click.echo(f"\nEpisode {episode_id}: {ep['title']!r}  [{ep['status']}]")

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
    params: list = [episode_id]
    if level:
        query += " AND w.cefr_level = ?"
        params.append(level)
    if status:
        query += " AND c.status = ?"
        params.append(status)
    query += " ORDER BY w.cefr_level, w.lemma"

    with db() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        msg = "No cards found."
        if ep["status"] not in ("cards_pending", "exported"):
            msg += " Run the cards stage first: speakyer run --stage cards"
        click.echo(f"  {msg}")
        return

    click.echo(f"  Cards: {len(rows)}\n")

    current_level = "UNSET"
    for row in rows:
        lev = row["cefr_level"] or "untagged"
        if lev != current_level:
            current_level = lev
            click.echo(f"  ── {lev} ──────────────────────────────────────────")

        sentence = (row["example_sentence"] or "").strip()
        front = _bold_word(sentence, row["surface_form"])
        if len(front) > 100:
            front = front[:97] + "..."

        ts = f"{row['start_time']:.1f}s" if row["start_time"] is not None else "?"
        status_tag = f"[{row['card_status']}]" if row["card_status"] != "pending" else ""

        click.echo(f"    [{row['card_id']}] {row['lemma']:<22} {row['pos']:<5} {lev:<3}  @{ts}  {status_tag}")
        click.echo(f"      Front: {front}")
        click.echo(
            f"      Back:  {row['lemma']} · {row['pos']}"
            + (f" · CEFR {row['cefr_level']}" if row["cefr_level"] else "")
            + f" · {row['source_name']}"
        )
        click.echo()


# ---------------------------------------------------------------------------
# speakyer db
# ---------------------------------------------------------------------------


@main.command()
def db() -> None:
    """Inspect the database: show table counts and recent rows."""
    from speakyer.config import config
    from speakyer.database import TABLES, get_connection, init

    def truncate(value: object, width: int) -> str:
        s = str(value) if value is not None else "NULL"
        return s[:width] if len(s) <= width else s[: width - 1] + "…"

    if not config.db_path.exists():
        click.echo(f"Database not found at {config.db_path} — initialising...")
        init()
        click.echo("Done.\n")

    conn = get_connection()
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    click.echo(f"Database: {config.db_path}\n")

    for table in TABLES:
        if table not in existing:
            click.echo(f"  {table}: (table not found)\n")
            continue

        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        bar = "─" * 52
        click.echo(bar)
        click.echo(f"  {table}  ({count} row{'s' if count != 1 else ''})")
        click.echo(bar)

        if count == 0:
            click.echo("  (empty)\n")
            continue

        cursor = conn.execute(f"SELECT * FROM {table} LIMIT 0")
        cols = [d[0] for d in cursor.description]
        col_w = min(16, max(len(c) for c in cols))
        widths = [max(len(c), col_w) for c in cols]

        header = "  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths))
        click.echo(header)
        click.echo("  " + "  ".join("-" * w for w in widths))

        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 3"
        ).fetchall()
        for row in reversed(rows):
            cells = [truncate(v, w).ljust(w) for v, w in zip(row, widths)]
            click.echo("  " + "  ".join(cells))
        click.echo()

    conn.close()


# ---------------------------------------------------------------------------
# speakyer seed-cefr
# ---------------------------------------------------------------------------


@main.command("seed-cefr")
@click.argument("csv_path", required=False, default=None, type=click.Path(exists=True))
def seed_cefr(csv_path: str | None) -> None:
    """Seed the cefr_words table from a CSV file.

    Defaults to data/cefr/de.csv if no path given.
    """
    import csv
    from pathlib import Path

    from speakyer.database import db as _db
    from speakyer.database import init

    VALID_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}
    default_csv = Path(__file__).parent.parent / "data" / "cefr" / "de.csv"
    path = Path(csv_path) if csv_path else default_csv

    if not path.exists():
        click.echo(f"Error: CSV file not found: {path}", err=True)
        sys.exit(1)

    rows: list[tuple[str, str, str | None]] = []
    skipped = 0

    with path.open(newline="", encoding="utf-8") as fh:
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
        click.echo("No valid rows found in CSV.")
        return

    init()

    with _db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO cefr_words (lemma, level, pos) VALUES (?, ?, ?)",
            rows,
        )

    click.echo(f"Seeded {len(rows)} CEFR words (skipped {skipped} invalid rows).")

    with _db() as conn:
        counts = conn.execute(
            "SELECT level, COUNT(*) FROM cefr_words GROUP BY level ORDER BY level"
        ).fetchall()
    for level, count in counts:
        click.echo(f"  {level}: {count}")


if __name__ == "__main__":
    main()
