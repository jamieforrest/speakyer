"""Tests for the NLP pipeline: NLPExtractor base, SpacyExtractor, and NLPStage."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from speakyer.database import db
from speakyer.nlp.base import NLPExtractor, Word
from speakyer.pipeline.base import PipelineContext
from speakyer.pipeline.stages import NLPStage
from speakyer.sources.base import Episode, SourceConfig

FAKE_SOURCE = SourceConfig(
    id=1,
    name="test",
    rss_url="https://example.com/feed.xml",
    language="de",
    transcript_strategy="whisper",
    active=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seed_episode_with_transcript(tmp_db: Path) -> tuple[Episode, int]:
    """Insert a source, episode, transcript and two segments. Return (episode, transcript_id)."""
    with db(tmp_db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'test', 'https://x.com')"
        )
        ep_cursor = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, status) "
            "VALUES (1, 'g1', 'Ep 1', 'transcribed')",
        )
        ep_id = ep_cursor.lastrowid

        tr_cursor = conn.execute(
            "INSERT INTO transcripts (episode_id, language_detected, duration_seconds) "
            "VALUES (?, 'de', 5.0)",
            (ep_id,),
        )
        tr_id = tr_cursor.lastrowid

        conn.executemany(
            "INSERT INTO transcript_segments (transcript_id, segment_index, text, start_time, end_time) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (tr_id, 0, "Guten Morgen Deutschland.", 0.0, 2.5),
                (tr_id, 1, "Die Nachrichten beginnen.", 2.5, 5.0),
            ],
        )

    ep = Episode(id=ep_id, guid="g1", source_id=1, title="Ep 1", status="transcribed")
    return ep, tr_id


def seed_cefr(tmp_db: Path, words: dict[str, str]) -> None:
    """Insert lemma→level pairs into cefr_words."""
    with db(tmp_db) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO cefr_words (lemma, level) VALUES (?, ?)",
            list(words.items()),
        )


class FakeExtractor(NLPExtractor):
    """Returns a fixed list of Words for every segment, keyed by segment text."""

    def __init__(self, words_by_text: dict[str, list[Word]] | None = None) -> None:
        self._words_by_text = words_by_text or {}
        self.calls: list[dict] = []

    def extract(
        self,
        text: str,
        episode_id: int,
        segment_id: int,
        start_time: float,
        cefr_lookup: dict[str, str],
    ) -> list[Word]:
        self.calls.append(
            {
                "text": text,
                "episode_id": episode_id,
                "segment_id": segment_id,
                "start_time": start_time,
                "cefr_lookup": cefr_lookup,
            }
        )
        return self._words_by_text.get(text, [])


def make_word(
    lemma: str,
    episode_id: int = 1,
    segment_id: int = 1,
    surface_form: str | None = None,
    pos: str = "NOUN",
    cefr_level: str | None = None,
    example_sentence: str = "",
    start_time: float = 0.0,
) -> Word:
    return Word(
        episode_id=episode_id,
        transcript_segment_id=segment_id,
        surface_form=surface_form or lemma.capitalize(),
        lemma=lemma,
        pos=pos,
        cefr_level=cefr_level,
        example_sentence=example_sentence,
        start_time=start_time,
    )


def run_stage(
    ep: Episode,
    tmp_db: Path,
    *,
    extractor: NLPExtractor | None = None,
    dry_run: bool = False,
) -> PipelineContext:
    ctx = PipelineContext(source_config=FAKE_SOURCE, episode=ep, dry_run=dry_run)
    with patch("speakyer.pipeline.stages.db", lambda: db(tmp_db)):
        return NLPStage(extractor=extractor or FakeExtractor()).run(ctx)


# ---------------------------------------------------------------------------
# Word dataclass
# ---------------------------------------------------------------------------


class TestWordDataclass:
    def test_fields_accessible(self):
        w = make_word("guten", cefr_level="A1", example_sentence="Guten Morgen.")
        assert w.lemma == "guten"
        assert w.cefr_level == "A1"
        assert w.example_sentence == "Guten Morgen."

    def test_segment_id_defaults_to_none(self):
        w = Word(
            episode_id=1,
            surface_form="Test",
            lemma="test",
            pos="NOUN",
            cefr_level=None,
            example_sentence="Test.",
            start_time=0.0,
        )
        assert w.transcript_segment_id is None


# ---------------------------------------------------------------------------
# NLPStage integration tests
# ---------------------------------------------------------------------------


class TestNLPStage:
    def test_extracts_words_and_writes_to_db(self, tmp_db):
        ep, tr_id = seed_episode_with_transcript(tmp_db)

        seg1_words = [make_word("morgen", episode_id=ep.id, segment_id=1, cefr_level="A1")]
        seg2_words = [make_word("nachricht", episode_id=ep.id, segment_id=2, cefr_level="A2")]

        extractor = FakeExtractor(
            {
                "Guten Morgen Deutschland.": seg1_words,
                "Die Nachrichten beginnen.": seg2_words,
            }
        )

        ctx = run_stage(ep, tmp_db, extractor=extractor)

        assert len(ctx.words) == 2
        assert ctx.words[0].lemma == "morgen"
        assert ctx.words[1].lemma == "nachricht"

        conn = sqlite3.connect(str(tmp_db))
        rows = conn.execute(
            "SELECT lemma, cefr_level FROM words WHERE episode_id = ?", (ep.id,)
        ).fetchall()
        conn.close()
        assert {r[0] for r in rows} == {"morgen", "nachricht"}
        assert {r[1] for r in rows} == {"A1", "A2"}

    def test_episode_status_updated_to_analyzed(self, tmp_db):
        ep, _ = seed_episode_with_transcript(tmp_db)
        run_stage(ep, tmp_db, extractor=FakeExtractor())

        conn = sqlite3.connect(str(tmp_db))
        status = conn.execute(
            "SELECT status FROM episodes WHERE id = ?", (ep.id,)
        ).fetchone()[0]
        conn.close()
        assert status == "analyzed"

    def test_idempotent_skips_if_words_exist(self, tmp_db):
        ep, _ = seed_episode_with_transcript(tmp_db)

        call_count = [0]

        class CountingExtractor(NLPExtractor):
            def extract(self, text, episode_id, segment_id, start_time, cefr_lookup):
                call_count[0] += 1
                return []

        # First run — processes segments.
        run_stage(ep, tmp_db, extractor=CountingExtractor())
        first_count = call_count[0]

        # Insert a dummy word so the idempotency check triggers.
        with db(tmp_db) as conn:
            conn.execute(
                "INSERT INTO words (episode_id, surface_form, lemma, pos, example_sentence) "
                "VALUES (?, 'Test', 'test', 'NOUN', 'Test.')",
                (ep.id,),
            )

        call_count[0] = 0
        run_stage(ep, tmp_db, extractor=CountingExtractor())
        # extractor must not be called again.
        assert call_count[0] == 0

    def test_skips_when_no_transcript(self, tmp_db):
        """Episode that was never transcribed should be silently skipped."""
        with db(tmp_db) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'test', 'https://x.com')"
            )
            cursor = conn.execute(
                "INSERT INTO episodes (source_id, guid, title, status) "
                "VALUES (1, 'g2', 'No Transcript', 'downloaded')"
            )
        ep = Episode(id=cursor.lastrowid, guid="g2", source_id=1, title="No Transcript")

        call_count = [0]

        class CountingExtractor(NLPExtractor):
            def extract(self, text, episode_id, segment_id, start_time, cefr_lookup):
                call_count[0] += 1
                return []

        ctx = run_stage(ep, tmp_db, extractor=CountingExtractor())

        assert call_count[0] == 0
        assert ctx.words == []

    def test_dry_run_makes_no_writes(self, tmp_db):
        ep, _ = seed_episode_with_transcript(tmp_db)

        call_count = [0]

        class CountingExtractor(NLPExtractor):
            def extract(self, text, episode_id, segment_id, start_time, cefr_lookup):
                call_count[0] += 1
                return [make_word("test", episode_id=episode_id)]

        ctx = run_stage(ep, tmp_db, extractor=CountingExtractor(), dry_run=True)

        assert call_count[0] == 0
        assert ctx.words == []

        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM words WHERE episode_id = ?", (ep.id,)
        ).fetchone()[0]
        status = conn.execute(
            "SELECT status FROM episodes WHERE id = ?", (ep.id,)
        ).fetchone()[0]
        conn.close()
        assert count == 0
        assert status == "transcribed"  # unchanged

    def test_cefr_lookup_passed_to_extractor(self, tmp_db):
        """CEFR lookup loaded from DB is forwarded to every extract() call."""
        ep, _ = seed_episode_with_transcript(tmp_db)
        seed_cefr(tmp_db, {"morgen": "A1", "nachricht": "A2"})

        received_lookups: list[dict] = []

        class RecordingExtractor(NLPExtractor):
            def extract(self, text, episode_id, segment_id, start_time, cefr_lookup):
                received_lookups.append(dict(cefr_lookup))
                return []

        run_stage(ep, tmp_db, extractor=RecordingExtractor())

        assert len(received_lookups) == 2  # one per segment
        for lookup in received_lookups:
            assert lookup.get("morgen") == "A1"
            assert lookup.get("nachricht") == "A2"

    def test_all_segment_texts_forwarded(self, tmp_db):
        ep, _ = seed_episode_with_transcript(tmp_db)

        extractor = FakeExtractor()
        run_stage(ep, tmp_db, extractor=extractor)

        texts = [c["text"] for c in extractor.calls]
        assert "Guten Morgen Deutschland." in texts
        assert "Die Nachrichten beginnen." in texts

    def test_words_with_no_cefr_match_have_null_level(self, tmp_db):
        ep, _ = seed_episode_with_transcript(tmp_db)

        # No CEFR seeds — all words should have cefr_level=None.
        word = make_word("unbekannt", episode_id=ep.id, cefr_level=None)
        extractor = FakeExtractor({"Guten Morgen Deutschland.": [word]})

        run_stage(ep, tmp_db, extractor=extractor)

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute(
            "SELECT cefr_level FROM words WHERE episode_id = ?", (ep.id,)
        ).fetchone()
        conn.close()
        assert row[0] is None


# ---------------------------------------------------------------------------
# SpacyExtractor (unit, no real model required)
# ---------------------------------------------------------------------------


class TestSpacyExtractor:
    def test_raises_import_error_when_spacy_unavailable(self, tmp_path):
        from speakyer.nlp.spacy_nlp import SpacyExtractor

        extractor = SpacyExtractor("some-model")
        with patch("speakyer.nlp.spacy_nlp._SPACY_AVAILABLE", False):
            with pytest.raises(ImportError, match="spacy is not installed"):
                _ = extractor.nlp

    def test_raises_os_error_when_model_missing(self):
        from unittest.mock import MagicMock, patch

        from speakyer.nlp.spacy_nlp import SpacyExtractor

        extractor = SpacyExtractor("nonexistent-model-xyz")

        fake_spacy = MagicMock()
        fake_spacy.load.side_effect = OSError("not found")

        with (
            patch("speakyer.nlp.spacy_nlp._SPACY_AVAILABLE", True),
            patch("speakyer.nlp.spacy_nlp.spacy", fake_spacy, create=True),
            patch.dict("sys.modules", {"spacy": fake_spacy}),
        ):
            # Reset cached instance so the property re-runs the load path.
            extractor._nlp = None
            with pytest.raises(OSError, match="nonexistent-model-xyz"):
                _ = extractor.nlp

    def test_extract_filters_to_content_pos(self):
        """Given a mocked spaCy doc, only NOUN/VERB/ADJ/ADV tokens are returned."""
        from unittest.mock import MagicMock

        from speakyer.nlp.spacy_nlp import SpacyExtractor

        extractor = SpacyExtractor("de_core_news_lg")

        # Build minimal mock tokens.
        def make_token(text, pos, lemma):
            t = MagicMock()
            t.text = text
            t.pos_ = pos
            t.lemma_ = lemma
            return t

        tokens = [
            make_token("Guten", "ADJ", "gut"),
            make_token("Morgen", "NOUN", "morgen"),
            make_token(",", "PUNCT", ","),
            make_token("die", "DET", "die"),
            make_token("Welt", "NOUN", "welt"),
        ]

        mock_nlp = MagicMock()
        mock_nlp.return_value = tokens
        extractor._nlp = mock_nlp

        words = extractor.extract(
            text="Guten Morgen, die Welt",
            episode_id=1,
            segment_id=10,
            start_time=1.0,
            cefr_lookup={"gut": "A1", "welt": "A1"},
        )

        assert len(words) == 3  # ADJ + NOUN + NOUN (PUNCT and DET excluded)
        lemmas = {w.lemma for w in words}
        assert lemmas == {"gut", "morgen", "welt"}
        assert all(w.episode_id == 1 for w in words)
        assert all(w.transcript_segment_id == 10 for w in words)
        assert all(w.start_time == 1.0 for w in words)

    def test_extract_lowercases_lemmas(self):
        from unittest.mock import MagicMock

        from speakyer.nlp.spacy_nlp import SpacyExtractor

        extractor = SpacyExtractor("de_core_news_lg")

        token = MagicMock()
        token.text = "Deutschland"
        token.pos_ = "NOUN"
        token.lemma_ = "Deutschland"  # spaCy sometimes keeps capitalisation

        mock_nlp = MagicMock()
        mock_nlp.return_value = [token]
        extractor._nlp = mock_nlp

        words = extractor.extract("Deutschland", 1, 1, 0.0, {})
        assert words[0].lemma == "deutschland"

    def test_extract_skips_single_char_tokens(self):
        from unittest.mock import MagicMock

        from speakyer.nlp.spacy_nlp import SpacyExtractor

        extractor = SpacyExtractor("de_core_news_lg")

        token = MagicMock()
        token.text = "I"
        token.pos_ = "NOUN"
        token.lemma_ = "I"

        mock_nlp = MagicMock()
        mock_nlp.return_value = [token]
        extractor._nlp = mock_nlp

        words = extractor.extract("I", 1, 1, 0.0, {})
        assert words == []
