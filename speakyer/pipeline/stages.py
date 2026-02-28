import json

import requests

from speakyer.config import config
from speakyer.database import db
from speakyer.pipeline.base import PipelineContext, PipelineStage
from speakyer.storage import LocalStorage
from speakyer.transcription.base import Transcript, Transcriber


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
        transcript = self.transcriber.transcribe(
            audio_path, language=ctx.source_config.language
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
