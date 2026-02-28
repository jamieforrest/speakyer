from pathlib import Path

from speakyer.transcription.base import Transcript, TranscriptSegment, Transcriber, WordTimestamp

# mlx-whisper is Apple Silicon only. Import lazily so the module can be
# loaded on any platform — the error surfaces only when transcribe() is called.
try:
    import mlx_whisper as _mlx_whisper

    _MLX_AVAILABLE = True
except ImportError:
    _mlx_whisper = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False


class LocalWhisperTranscriber(Transcriber):
    def __init__(self, model: str) -> None:
        self.model = model

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        if not _MLX_AVAILABLE:
            raise ImportError(
                "mlx-whisper is not installed. "
                "Install it with: pip install mlx-whisper  (requires Apple Silicon)"
            )

        result = _mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model,
            word_timestamps=True,
            language=language,
        )
        return self._parse_result(result)

    def _parse_result(self, result: dict) -> Transcript:
        raw_segments = result.get("segments") or []
        duration = raw_segments[-1]["end"] if raw_segments else 0.0

        segments = [
            TranscriptSegment(
                index=i,
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                words=[
                    WordTimestamp(
                        word=w["word"].strip(),
                        start=w["start"],
                        end=w["end"],
                        probability=w.get("probability", 0.0),
                    )
                    for w in seg.get("words", [])
                ],
            )
            for i, seg in enumerate(raw_segments)
            if seg["text"].strip() and seg["start"] < seg["end"]
        ]

        return Transcript(
            text=result.get("text", "").strip(),
            language=result.get("language"),
            duration=duration,
            segments=segments,
        )
