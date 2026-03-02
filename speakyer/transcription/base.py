from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float = 0.0


@dataclass
class TranscriptSegment:
    index: int
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class Transcript:
    text: str
    language: str | None
    duration: float
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "index": s.index,
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        segments = [
            TranscriptSegment(
                index=s["index"],
                text=s["text"],
                start=s["start"],
                end=s["end"],
                words=[
                    WordTimestamp(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        probability=w.get("probability", 0.0),
                    )
                    for w in s.get("words", [])
                ],
            )
            for s in data["segments"]
        ]
        return cls(
            text=data["text"],
            language=data.get("language"),
            duration=data["duration"],
            segments=segments,
        )


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript: ...
