from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from speakyer.sources.base import Episode, SourceConfig

if TYPE_CHECKING:
    from speakyer.nlp.base import Word
    from speakyer.transcription.base import Transcript


@dataclass
class PipelineContext:
    source_config: SourceConfig
    episode: Episode
    dry_run: bool = False
    audio_path: Path | None = None
    transcript: Transcript | None = None  # set by TranscribeStage
    words: list[Word] = field(default_factory=list)  # set by NLPStage
    # cards: list[Card] = field(...)       # Milestone 4


class PipelineStage(ABC):
    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
