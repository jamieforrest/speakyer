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


def get_known_guids(source_id: int) -> set[str]:
    with db() as conn:
        rows = conn.execute(
            "SELECT guid FROM episodes WHERE source_id = ?", (source_id,)
        ).fetchall()
    return {row["guid"] for row in rows}


def insert_episodes(episodes: list[Episode]) -> list[Episode]:
    """Persist new episodes, skipping any whose GUID already exists.

    Returns only the newly inserted episodes, with their DB id populated.
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
                    ep.published_at.isoformat() if ep.published_at else None,
                    ep.audio_url,
                    ep.status,
                ),
            )
            if cursor.rowcount:
                ep.id = cursor.lastrowid
                inserted.append(ep)
    return inserted
