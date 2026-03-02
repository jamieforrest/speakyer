from datetime import datetime

import feedparser

from speakyer.sources.base import ContentSource, Episode, SourceConfig


class RSSPodcastSource(ContentSource):
    def get_new_episodes(
        self, source: SourceConfig, known_guids: set[str]
    ) -> list[Episode]:
        feed = feedparser.parse(source.rss_url)
        episodes = []

        for entry in feed.entries:
            guid = entry.get("id") or entry.get("link", "")
            if guid in known_guids:
                continue

            audio_url = None
            for enc in entry.get("enclosures", []):
                if enc.get("type", "").startswith("audio/"):
                    audio_url = enc.get("href") or enc.get("url")
                    break

            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published_at = datetime(*entry.published_parsed[:6])
                except (TypeError, ValueError):
                    pass

            episodes.append(
                Episode(
                    guid=guid,
                    source_id=source.id,
                    title=entry.get("title"),
                    published_at=published_at,
                    audio_url=audio_url,
                )
            )

        return episodes
