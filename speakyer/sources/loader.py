import yaml

from speakyer.config import config
from speakyer.database import db
from speakyer.sources.base import Episode, SourceConfig


def sync_sources() -> list[SourceConfig]:
    """Upsert sources from sources.yaml and return all active SourceConfigs."""
    if config.sources_yaml.exists():
        with open(config.sources_yaml) as f:
            data = yaml.safe_load(f) or {}

        with db() as conn:
            for s in data.get("sources", []):
                conn.execute(
                    """
                    INSERT INTO sources (name, rss_url, language, transcript_strategy, active)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        rss_url = excluded.rss_url,
                        language = excluded.language,
                        active = excluded.active
                    """,
                    (
                        s["name"],
                        s["rss_url"],
                        s.get("language", "de"),
                        s.get("transcript_strategy", "whisper"),
                        1 if s.get("active", True) else 0,
                    ),
                )

    return _fetch_active_sources()


def _fetch_active_sources() -> list[SourceConfig]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, name, rss_url, language, transcript_strategy, active
            FROM sources WHERE active = 1
            """
        ).fetchall()
    return [
        SourceConfig(
            id=row["id"],
            name=row["name"],
            rss_url=row["rss_url"],
            language=row["language"],
            transcript_strategy=row["transcript_strategy"],
            active=bool(row["active"]),
        )
        for row in rows
    ]


def get_pending_episodes(source_id: int, statuses: set[str]) -> list[Episode]:
    """Return episodes for a source whose status is in the given set."""
    placeholders = ",".join("?" * len(statuses))
    with db() as conn:
        rows = conn.execute(
            f"SELECT id, guid, title, published_at, audio_url, audio_path, status "
            f"FROM episodes WHERE source_id = ? AND status IN ({placeholders})",
            (source_id, *statuses),
        ).fetchall()
    return [
        Episode(
            id=row["id"],
            guid=row["guid"],
            source_id=source_id,
            title=row["title"],
            published_at=row["published_at"],
            audio_url=row["audio_url"],
            audio_path=row["audio_path"],
            status=row["status"],
        )
        for row in rows
    ]


def get_episode_by_id(episode_id: int) -> tuple[Episode, SourceConfig] | None:
    """Return (Episode, SourceConfig) for a given episode ID, or None if not found."""
    with db() as conn:
        row = conn.execute(
            "SELECT e.id, e.guid, e.title, e.published_at, e.audio_url, e.audio_path, e.status, "
            "e.source_id, s.name, s.rss_url, s.language, s.transcript_strategy, s.active "
            "FROM episodes e JOIN sources s ON s.id = e.source_id WHERE e.id = ?",
            (episode_id,),
        ).fetchone()
    if not row:
        return None
    ep = Episode(
        id=row["id"],
        guid=row["guid"],
        source_id=row["source_id"],
        title=row["title"],
        published_at=row["published_at"],
        audio_url=row["audio_url"],
        audio_path=row["audio_path"],
        status=row["status"],
    )
    src = SourceConfig(
        id=row["source_id"],
        name=row["name"],
        rss_url=row["rss_url"],
        language=row["language"],
        transcript_strategy=row["transcript_strategy"],
        active=bool(row["active"]),
    )
    return ep, src


def get_known_guids(source_id: int) -> set[str]:
    with db() as conn:
        rows = conn.execute(
            "SELECT guid FROM episodes WHERE source_id = ?", (source_id,)
        ).fetchall()
    return {row["guid"] for row in rows}


def insert_episodes(episodes: list[Episode]) -> list[Episode]:
    """Persist new episodes, skipping any whose GUID already exists.

    Returns only the newly inserted episodes, with their DB id populated.
    The datetime adapter registered in database.py handles published_at formatting.
    """
    inserted = []
    with db() as conn:
        for ep in episodes:
            cursor = conn.execute(
                """
                INSERT INTO episodes (source_id, guid, title, published_at, audio_url, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(guid) DO NOTHING
                """,
                (
                    ep.source_id,
                    ep.guid,
                    ep.title,
                    ep.published_at,  # passed as datetime; adapter formats it
                    ep.audio_url,
                    ep.status,
                ),
            )
            if cursor.rowcount:
                ep.id = cursor.lastrowid
                inserted.append(ep)
    return inserted
