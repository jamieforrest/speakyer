"""Tests for source configuration, RSS parsing, and episode persistence."""

import textwrap
from datetime import datetime
from pathlib import Path

import feedparser
import pytest

from speakyer.database import db
from speakyer.sources.base import Episode, SourceConfig
from speakyer.sources.loader import get_known_guids, insert_episodes, sync_sources
from speakyer.sources.rss import RSSPodcastSource

# ---------------------------------------------------------------------------
# Test data — parsed once at import time so mocks don't interfere
# ---------------------------------------------------------------------------

FAKE_RSS = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Test Podcast</title>
        <item>
          <title>Episode 1</title>
          <guid>https://example.com/ep1</guid>
          <pubDate>Mon, 01 Jan 2024 20:00:00 +0000</pubDate>
          <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg" length="1234"/>
        </item>
        <item>
          <title>Episode 2</title>
          <guid>https://example.com/ep2</guid>
          <pubDate>Tue, 02 Jan 2024 20:00:00 +0000</pubDate>
          <enclosure url="https://example.com/ep2.mp3" type="audio/mpeg" length="5678"/>
        </item>
      </channel>
    </rss>
    """
)

FAKE_RSS_NO_EP1_AUDIO = FAKE_RSS.replace(
    '<enclosure url="https://example.com/ep1.mp3" type="audio/mpeg" length="1234"/>',
    "",
)

# Parse feeds with the real feedparser before any mocking occurs.
_PARSED_FEED = feedparser.parse(FAKE_RSS)
_PARSED_FEED_NO_EP1_AUDIO = feedparser.parse(FAKE_RSS_NO_EP1_AUDIO)

FAKE_SOURCE = SourceConfig(
    id=1,
    name="test",
    rss_url="https://example.com/feed.xml",
    language="de",
    transcript_strategy="whisper",
    active=True,
)


def seed_source(tmp_db: Path) -> SourceConfig:
    with db(tmp_db) as conn:
        cursor = conn.execute(
            "INSERT INTO sources (name, rss_url) VALUES (?, ?)",
            ("test", "https://example.com/feed.xml"),
        )
    return SourceConfig(
        id=cursor.lastrowid,
        name="test",
        rss_url="https://example.com/feed.xml",
        language="de",
        transcript_strategy="whisper",
        active=True,
    )


# ---------------------------------------------------------------------------
# RSSPodcastSource
# Patch speakyer.sources.rss.feedparser (where it's used), not the global.
# ---------------------------------------------------------------------------


class TestRSSPodcastSource:
    def test_parses_episodes_from_feed(self, mocker):
        mocker.patch("speakyer.sources.rss.feedparser").parse.return_value = _PARSED_FEED
        episodes = RSSPodcastSource().get_new_episodes(FAKE_SOURCE, set())

        assert len(episodes) == 2
        assert episodes[0].guid == "https://example.com/ep1"
        assert episodes[0].title == "Episode 1"
        assert episodes[0].audio_url == "https://example.com/ep1.mp3"
        assert isinstance(episodes[0].published_at, datetime)

    def test_deduplicates_known_guids(self, mocker):
        mocker.patch("speakyer.sources.rss.feedparser").parse.return_value = _PARSED_FEED
        episodes = RSSPodcastSource().get_new_episodes(
            FAKE_SOURCE, {"https://example.com/ep1"}
        )

        assert len(episodes) == 1
        assert episodes[0].guid == "https://example.com/ep2"

    def test_skips_entry_without_audio_enclosure(self, mocker):
        mocker.patch(
            "speakyer.sources.rss.feedparser"
        ).parse.return_value = _PARSED_FEED_NO_EP1_AUDIO
        episodes = RSSPodcastSource().get_new_episodes(FAKE_SOURCE, set())

        ep1 = next(e for e in episodes if e.guid == "https://example.com/ep1")
        assert ep1.audio_url is None

    def test_all_known_guids_returns_empty(self, mocker):
        mocker.patch("speakyer.sources.rss.feedparser").parse.return_value = _PARSED_FEED
        episodes = RSSPodcastSource().get_new_episodes(
            FAKE_SOURCE,
            {"https://example.com/ep1", "https://example.com/ep2"},
        )
        assert episodes == []


# ---------------------------------------------------------------------------
# insert_episodes / get_known_guids
# patched_config fixture (conftest.py) redirects config.db_path globally.
# ---------------------------------------------------------------------------


class TestInsertEpisodes:
    def test_inserts_new_episodes_and_returns_with_ids(self, patched_config):
        source = seed_source(patched_config.db_path)
        episodes = [
            Episode(guid="guid-1", source_id=source.id, title="Ep 1"),
            Episode(guid="guid-2", source_id=source.id, title="Ep 2"),
        ]
        inserted = insert_episodes(episodes)

        assert len(inserted) == 2
        assert all(ep.id is not None for ep in inserted)

    def test_skips_duplicate_guid(self, patched_config):
        source = seed_source(patched_config.db_path)
        ep = Episode(guid="guid-1", source_id=source.id, title="Ep 1")

        first = insert_episodes([ep])
        assert len(first) == 1

        duplicate = Episode(guid="guid-1", source_id=source.id, title="Duplicate")
        second = insert_episodes([duplicate])
        assert len(second) == 0

    def test_get_known_guids_returns_all_guids_for_source(self, patched_config):
        source = seed_source(patched_config.db_path)
        insert_episodes([
            Episode(guid="guid-a", source_id=source.id),
            Episode(guid="guid-b", source_id=source.id),
        ])
        known = get_known_guids(source.id)
        assert known == {"guid-a", "guid-b"}


# ---------------------------------------------------------------------------
# sync_sources
# ---------------------------------------------------------------------------


class TestSyncSources:
    def test_seeds_db_from_yaml(self, patched_config):
        patched_config.sources_yaml.write_text(
            "sources:\n"
            "  - name: testcast\n"
            "    rss_url: https://example.com/rss\n"
            "    language: de\n"
            "    active: true\n"
        )
        sources = sync_sources()

        assert len(sources) == 1
        assert sources[0].name == "testcast"
        assert sources[0].rss_url == "https://example.com/rss"

    def test_updates_existing_source_on_url_change(self, patched_config):
        patched_config.sources_yaml.write_text(
            "sources:\n"
            "  - name: testcast\n"
            "    rss_url: https://example.com/rss-v1\n"
            "    active: true\n"
        )
        sync_sources()

        patched_config.sources_yaml.write_text(
            "sources:\n"
            "  - name: testcast\n"
            "    rss_url: https://example.com/rss-v2\n"
            "    active: true\n"
        )
        sources = sync_sources()
        assert sources[0].rss_url == "https://example.com/rss-v2"

    def test_inactive_source_excluded_from_results(self, patched_config):
        patched_config.sources_yaml.write_text(
            "sources:\n"
            "  - name: active-cast\n"
            "    rss_url: https://example.com/active\n"
            "    active: true\n"
            "  - name: inactive-cast\n"
            "    rss_url: https://example.com/inactive\n"
            "    active: false\n"
        )
        sources = sync_sources()
        names = [s.name for s in sources]

        assert "active-cast" in names
        assert "inactive-cast" not in names
