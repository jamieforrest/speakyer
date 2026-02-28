import json
import threading
import time
from collections import Counter

import requests

from speakyer.cards.base import Card
from speakyer.cards.anki_connect import AnkiConnectExporter, _invoke
from speakyer.config import config
from speakyer.database import db
from speakyer.exporters.base import CardExporter
from speakyer.nlp.audio_clipper import AudioClipper
from speakyer.nlp.base import NLPExtractor, Word
from speakyer.pipeline.base import PipelineContext, PipelineStage
from speakyer.storage import LocalStorage
from speakyer.transcription.base import Transcript, Transcriber


def _transcribe_with_heartbeat(
    transcriber: Transcriber,
    audio_path,
    *,
    language: str | None = None,
    interval: int = 15,
) -> "Transcript":
    """Call transcriber.transcribe() on a background thread, printing elapsed-time
    heartbeats every *interval* seconds so the terminal doesn't look frozen."""
    result: list = []
    error: list = []

    def _run():
        try:
            result.append(transcriber.transcribe(audio_path, language=language))
        except Exception as exc:  # noqa: BLE001
            error.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    elapsed = 0
    while t.is_alive():
        t.join(timeout=interval)
        elapsed += interval
        if t.is_alive():
            mins, secs = divmod(elapsed, 60)
            print(f"  ... still transcribing ({mins}:{secs:02d} elapsed)")

    if error:
        raise error[0]
    return result[0]


class DownloadStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> PipelineContext:
        ep = ctx.episode

        if not ep.audio_url:
            print(f"  [skip] {ep.title!r}: no audio URL")
            return ctx

        storage = LocalStorage(config.data_dir)
        rel_path = f"audio/{ep.id}.mp3"

        if storage.exists(rel_path):
            print(f"  [skip] {ep.title!r}: already downloaded")
            ctx.audio_path = storage.absolute_path(rel_path)
            return ctx

        if ctx.dry_run:
            print(f"  [dry-run] would download: {ep.title!r}")
            return ctx

        print(f"  Downloading: {ep.title!r} ...")
        with requests.get(ep.audio_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            audio_data = b"".join(response.iter_content(chunk_size=65536))

        storage.write(rel_path, audio_data)
        ctx.audio_path = storage.absolute_path(rel_path)

        with db() as conn:
            conn.execute(
                "UPDATE episodes SET audio_path = ?, status = ? WHERE id = ?",
                (rel_path, "downloaded", ep.id),
            )

        size_mb = len(audio_data) / 1_048_576
        print(f"  Saved {size_mb:.1f} MB → {ctx.audio_path}")
        return ctx


class TranscribeStage(PipelineStage):
    def __init__(self, transcriber: Transcriber | None = None) -> None:
        # Injected transcriber is used as-is (enables testing without mlx-whisper).
        # When None, a LocalWhisperTranscriber is created on first use.
        self._transcriber = transcriber

    @property
    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            from speakyer.transcription.whisper_local import LocalWhisperTranscriber

            self._transcriber = LocalWhisperTranscriber(config.whisper_model)
        return self._transcriber

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ep = ctx.episode
        storage = LocalStorage(config.data_dir)

        # Idempotency: skip if transcript already exists in DB.
        with db() as conn:
            existing = conn.execute(
                "SELECT id FROM transcripts WHERE episode_id = ?", (ep.id,)
            ).fetchone()
        if existing:
            print(f"  [skip] {ep.title!r}: already transcribed")
            return ctx

        # Resolve audio path: prefer ctx (set by DownloadStage), fall back to DB value.
        audio_path = ctx.audio_path
        if audio_path is None and ep.audio_path:
            audio_path = storage.absolute_path(ep.audio_path)

        if audio_path is None or not audio_path.exists():
            print(f"  [skip] {ep.title!r}: audio file not found — run download stage first")
            return ctx

        if ctx.dry_run:
            print(f"  [dry-run] would transcribe: {ep.title!r}")
            return ctx

        print(f"  Transcribing: {ep.title!r} ...")
        transcript = _transcribe_with_heartbeat(
            self.transcriber, audio_path, language=ctx.source_config.language
        )

        # Persist raw JSON (needed for audio clip extraction in v1.1).
        rel_path = f"transcripts/{ep.id}.json"
        storage.write(rel_path, json.dumps(transcript.to_dict()).encode())

        # Write to DB.
        with db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO transcripts (episode_id, raw_json_path, language_detected, duration_seconds)
                VALUES (?, ?, ?, ?)
                """,
                (ep.id, rel_path, transcript.language, transcript.duration),
            )
            transcript_id = cursor.lastrowid

            conn.executemany(
                """
                INSERT INTO transcript_segments (transcript_id, segment_index, text, start_time, end_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (transcript_id, seg.index, seg.text, seg.start, seg.end)
                    for seg in transcript.segments
                ],
            )

            conn.execute(
                "UPDATE episodes SET status = ? WHERE id = ?",
                ("transcribed", ep.id),
            )

        ctx.transcript = transcript
        mins, secs = divmod(int(transcript.duration), 60)
        print(
            f"  Transcribed {len(transcript.segments)} segments "
            f"({mins}:{secs:02d}) — lang: {transcript.language}"
        )
        return ctx


class NLPStage(PipelineStage):
    """Extract vocabulary from transcript segments and look up CEFR levels.

    Words are stored in the ``words`` table, one row per token per segment.
    Episode status advances to ``"analyzed"`` after completion.

    An injectable *extractor* enables testing without a real spaCy model.
    """

    def __init__(self, extractor: NLPExtractor | None = None) -> None:
        self._extractor = extractor

    @property
    def extractor(self) -> NLPExtractor:
        if self._extractor is None:
            from speakyer.nlp.spacy_nlp import SpacyExtractor

            self._extractor = SpacyExtractor(config.spacy_model)
        return self._extractor

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ep = ctx.episode

        # Idempotency: skip if words already exist in DB for this episode.
        with db() as conn:
            existing_count = conn.execute(
                "SELECT COUNT(*) FROM words WHERE episode_id = ?", (ep.id,)
            ).fetchone()[0]
        if existing_count:
            print(f"  [skip] {ep.title!r}: already analyzed ({existing_count} words)")
            return ctx

        # Load transcript segments from DB.
        with db() as conn:
            segments = conn.execute(
                """
                SELECT ts.id, ts.text, ts.start_time
                FROM transcript_segments ts
                JOIN transcripts t ON ts.transcript_id = t.id
                WHERE t.episode_id = ?
                ORDER BY ts.segment_index
                """,
                (ep.id,),
            ).fetchall()

        if not segments:
            print(f"  [skip] {ep.title!r}: no transcript found — run transcribe stage first")
            return ctx

        if ctx.dry_run:
            print(f"  [dry-run] would analyze: {ep.title!r} ({len(segments)} segments)")
            return ctx

        # Pre-load CEFR lookup table into memory (one query for all words).
        with db() as conn:
            cefr_rows = conn.execute("SELECT lemma, level FROM cefr_words").fetchall()
        cefr_lookup: dict[str, str] = {row["lemma"]: row["level"] for row in cefr_rows}

        # Extract words from each segment.
        all_words: list[Word] = []
        for seg in segments:
            words = self.extractor.extract(
                text=seg["text"],
                episode_id=ep.id,
                segment_id=seg["id"],
                start_time=seg["start_time"],
                cefr_lookup=cefr_lookup,
            )
            all_words.extend(words)

        # Persist to DB.
        with db() as conn:
            conn.executemany(
                """
                INSERT INTO words
                    (episode_id, transcript_segment_id, surface_form, lemma,
                     pos, cefr_level, example_sentence, start_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        w.episode_id,
                        w.transcript_segment_id,
                        w.surface_form,
                        w.lemma,
                        w.pos,
                        w.cefr_level,
                        w.example_sentence,
                        w.start_time,
                    )
                    for w in all_words
                ],
            )
            conn.execute(
                "UPDATE episodes SET status = ? WHERE id = ?",
                ("analyzed", ep.id),
            )

        ctx.words = all_words

        cefr_counts = Counter(w.cefr_level for w in all_words if w.cefr_level)
        tagged = sum(cefr_counts.values())
        levels_str = " ".join(f"{lvl}:{n}" for lvl, n in sorted(cefr_counts.items()))
        print(
            f"  Extracted {len(all_words)} words "
            f"({tagged} with CEFR level"
            + (f": {levels_str}" if levels_str else "")
            + ")"
        )
        return ctx


class CardGenStage(PipelineStage):
    """Create one Anki card per unique lemma not yet queued for export.

    Deduplication is global: if a lemma already has a card from any episode it
    is skipped.  The chosen word row (earliest occurrence in this episode) acts
    as the canonical example sentence for the card.

    Card content (front/back HTML, tags) is derived at export time (Milestone 5);
    this stage only decides *which* words become cards and writes the queue rows.
    """

    DECK = "Speakyer"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ep = ctx.episode

        # Idempotency: skip if cards already exist for this episode's words.
        with db() as conn:
            existing_count = conn.execute(
                """
                SELECT COUNT(*) FROM cards c
                JOIN words w ON c.word_id = w.id
                WHERE w.episode_id = ?
                """,
                (ep.id,),
            ).fetchone()[0]
        if existing_count:
            print(f"  [skip] {ep.title!r}: cards already generated ({existing_count} cards)")
            return ctx

        # Load words for this episode (one row per token; ordered by appearance).
        with db() as conn:
            word_rows = conn.execute(
                """
                SELECT id, lemma FROM words
                WHERE episode_id = ?
                ORDER BY id
                """,
                (ep.id,),
            ).fetchall()

        if not word_rows:
            print(f"  [skip] {ep.title!r}: no words found — run NLP stage first")
            return ctx

        if ctx.dry_run:
            print(f"  [dry-run] would generate cards for: {ep.title!r} ({len(word_rows)} words)")
            return ctx

        # Fetch lemmas that already have cards anywhere in the DB.
        with db() as conn:
            existing_lemmas: set[str] = {
                row["lemma"]
                for row in conn.execute(
                    "SELECT DISTINCT w.lemma FROM words w JOIN cards c ON c.word_id = w.id"
                ).fetchall()
            }

        # Pick the first word_id for each new lemma (earliest segment occurrence).
        new_cards: list[Card] = []
        seen: set[str] = set()
        for row in word_rows:
            lemma = row["lemma"]
            if lemma in existing_lemmas or lemma in seen:
                continue
            seen.add(lemma)
            new_cards.append(Card(word_id=row["id"], deck_name=self.DECK))

        with db() as conn:
            cursors = conn.executemany(
                "INSERT INTO cards (word_id, deck_name) VALUES (?, ?)",
                [(c.word_id, c.deck_name) for c in new_cards],
            )
            conn.execute(
                "UPDATE episodes SET status = ? WHERE id = ?",
                ("cards_pending", ep.id),
            )

        ctx.cards = new_cards
        skipped = len(word_rows) - len({r["lemma"] for r in word_rows}) + (
            len({r["lemma"] for r in word_rows}) - len(new_cards)
        )
        print(
            f"  Generated {len(new_cards)} new card(s) "
            f"({skipped} duplicate/existing lemmas skipped)"
        )
        return ctx


class ExportStage(PipelineStage):
    """Push pending cards to Anki via AnkiConnect and mark them exported.

    An injectable *exporter* enables testing without a running Anki instance.
    When *exporter* is ``None`` an :class:`AnkiConnectExporter` is created on
    first use (connects to ``http://localhost:8765`` by default).
    """

    def __init__(self, exporter: CardExporter | None = None) -> None:
        self._exporter = exporter

    @property
    def exporter(self) -> CardExporter:
        if self._exporter is None:
            self._exporter = AnkiConnectExporter()
        return self._exporter

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ep = ctx.episode

        # Idempotency: skip if all cards for this episode are already exported.
        with db() as conn:
            pending_count = conn.execute(
                """
                SELECT COUNT(*) FROM cards c
                JOIN words w ON c.word_id = w.id
                WHERE w.episode_id = ? AND c.status = 'pending'
                """,
                (ep.id,),
            ).fetchone()[0]

        if not pending_count:
            print(f"  [skip] {ep.title!r}: no pending cards to export")
            return ctx

        # Load enriched card rows (all data needed to build Anki notes).
        with db() as conn:
            rows = conn.execute(
                """
                SELECT c.id AS card_id, c.word_id, c.deck_name,
                       c.audio_clip_path,
                       w.lemma, w.surface_form, w.pos, w.cefr_level,
                       w.example_sentence,
                       e.title AS episode_title, e.published_at,
                       s.name AS source_name
                FROM cards c
                JOIN words w ON c.word_id = w.id
                JOIN episodes e ON w.episode_id = e.id
                JOIN sources s ON e.source_id = s.id
                WHERE w.episode_id = ? AND c.status = 'pending'
                ORDER BY c.id
                """,
                (ep.id,),
            ).fetchall()

        if ctx.dry_run:
            print(f"  [dry-run] would export {len(rows)} card(s) for: {ep.title!r}")
            return ctx

        # Convert to dicts and resolve audio clip paths to absolute before export.
        storage = LocalStorage(config.data_dir)
        enriched: list[dict] = []
        for row in rows:
            d = dict(row)
            if d["audio_clip_path"]:
                d["audio_clip_abs_path"] = str(storage.absolute_path(d["audio_clip_path"]))
            enriched.append(d)

        print(f"  Exporting {len(enriched)} card(s) to Anki ...")
        try:
            note_ids = self.exporter.export_rows(enriched)
        except ConnectionError as exc:
            print(f"  [error] {exc}")
            return ctx

        # Persist results: store note IDs, mark cards exported.
        exported_card_ids = []
        failed = 0
        for row, note_id in zip(rows, note_ids):
            if note_id:
                exported_card_ids.append((note_id, row["card_id"]))
            else:
                failed += 1

        with db() as conn:
            conn.executemany(
                """
                UPDATE cards
                SET anki_note_id = ?, status = 'exported',
                    exported_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                exported_card_ids,
            )
            if failed:
                # Mark rejected notes as failed so they don't block re-runs.
                failed_ids = [
                    (row["card_id"],)
                    for row, nid in zip(rows, note_ids)
                    if not nid
                ]
                conn.executemany(
                    "UPDATE cards SET status = 'failed' WHERE id = ?",
                    failed_ids,
                )
            conn.execute(
                "UPDATE episodes SET status = 'exported' WHERE id = ?",
                (ep.id,),
            )

        exported = len(exported_card_ids)
        msg = f"  Exported {exported} card(s) to Anki"
        if failed:
            msg += f" ({failed} rejected — likely duplicates already in Anki)"
        print(msg)
        return ctx


class AudioClipStage(PipelineStage):
    """Extract audio clips for all pending cards belonging to an episode.

    For each card whose ``audio_clip_path`` is NULL, calls
    :class:`~speakyer.nlp.audio_clipper.AudioClipper` to extract a padded
    MP3 segment and stores the relative path in ``cards.audio_clip_path``.

    An injectable *clipper* enables testing without real audio files.
    """

    def __init__(self, clipper: AudioClipper | None = None) -> None:
        self._clipper = clipper

    @property
    def clipper(self) -> AudioClipper:
        if self._clipper is None:
            self._clipper = AudioClipper()
        return self._clipper

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ep = ctx.episode

        with db() as conn:
            rows = conn.execute(
                """
                SELECT c.id AS card_id, c.word_id
                FROM cards c
                JOIN words w ON c.word_id = w.id
                WHERE w.episode_id = ? AND c.audio_clip_path IS NULL
                  AND c.status IN ('pending', 'exported')
                ORDER BY c.id
                """,
                (ep.id,),
            ).fetchall()

        if not rows:
            print(f"  [skip] {ep.title!r}: no cards need audio clips")
            return ctx

        if ctx.dry_run:
            print(f"  [dry-run] would extract {len(rows)} audio clip(s) for: {ep.title!r}")
            return ctx

        total = len(rows)
        print(f"  Extracting {total} audio clip(s) for: {ep.title!r} ...")
        extracted = skipped = 0
        for i, row in enumerate(rows, 1):
            clip_path = self.clipper.extract(row["word_id"])
            if clip_path is None:
                skipped += 1
            else:
                rel = str(clip_path.relative_to(self.clipper.storage.base_dir))
                with db() as conn:
                    conn.execute(
                        "UPDATE cards SET audio_clip_path = ? WHERE id = ?",
                        (rel, row["card_id"]),
                    )
                extracted += 1
            if i % 50 == 0 or i == total:
                print(f"    ... {i}/{total} ({skipped} skipped)", flush=True)

        msg = f"  Extracted {extracted} audio clip(s)"
        if skipped:
            msg += f" ({skipped} skipped — missing transcript or audio)"
        print(msg)
        return ctx


class CardUpdateStage(PipelineStage):
    """Push audio clips to existing Anki notes via AnkiConnect updateNote.

    Processes exported cards for the episode that have a clip path but have
    not yet been updated in Anki (``status = 'exported'``).  After a
    successful update, the card status advances to ``'audio_updated'``.

    An injectable *url* enables testing without a running Anki instance.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url

    @property
    def url(self) -> str:
        return self._url or config.anki_connect_url

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ep = ctx.episode

        with db() as conn:
            rows = conn.execute(
                """
                SELECT c.id AS card_id, c.anki_note_id, c.audio_clip_path,
                       w.lemma, w.surface_form, w.pos, w.cefr_level,
                       w.example_sentence, c.deck_name,
                       e.title AS episode_title,
                       s.name AS source_name
                FROM cards c
                JOIN words w ON c.word_id = w.id
                JOIN episodes e ON w.episode_id = e.id
                JOIN sources s ON e.source_id = s.id
                WHERE w.episode_id = ?
                  AND c.anki_note_id IS NOT NULL
                  AND c.audio_clip_path IS NOT NULL
                  AND c.status = 'exported'
                ORDER BY c.id
                """,
                (ep.id,),
            ).fetchall()

        if not rows:
            print(f"  [skip] {ep.title!r}: no cards ready for audio update")
            return ctx

        if ctx.dry_run:
            print(f"  [dry-run] would update {len(rows)} Anki note(s) for: {ep.title!r}")
            return ctx

        storage = LocalStorage(config.data_dir)
        print(f"  Updating {len(rows)} Anki note(s) with audio ...")
        updated = failed = 0

        for row in rows:
            try:
                abs_clip = storage.absolute_path(row["audio_clip_path"])
                filename = f"speakyer_{row['anki_note_id']}.mp3"
                note_payload = {
                    "id": row["anki_note_id"],
                    "fields": {
                        "Front": AnkiConnectExporter._build_front(row),
                        "Back": AnkiConnectExporter._build_back(row),
                    },
                    "audio": [
                        {
                            "path": str(abs_clip),
                            "filename": filename,
                            "fields": ["Back"],
                        }
                    ],
                }
                _invoke("updateNote", self.url, note=note_payload)
                with db() as conn:
                    conn.execute(
                        "UPDATE cards SET status = 'audio_updated' WHERE id = ?",
                        (row["card_id"],),
                    )
                updated += 1
            except ConnectionError as exc:
                print(f"  [error] {exc}")
                return ctx
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] note {row['anki_note_id']} failed: {exc}")
                failed += 1

        msg = f"  Updated {updated} note(s) with audio"
        if failed:
            msg += f" ({failed} failed)"
        print(msg)
        return ctx


# Auxiliary pipeline for adding audio to already-exported cards.
AUDIO_UPDATE_PIPELINE: list[type[PipelineStage]] = [AudioClipStage, CardUpdateStage]
