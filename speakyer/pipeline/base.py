from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from speakyer.sources.base import Episode, SourceConfig


@dataclass
class PipelineContext:
    source_config: SourceConfig
    episode: Episode
    dry_run: bool = False
    audio_path: Path | None = None
    # Populated by later stages:
    # transcript: Transcript | None = None   # Milestone 2
    # words: list[Word] = field(...)         # Milestone 3
    # cards: list[Card] = field(...)         # Milestone 4


class PipelineStage(ABC):
    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
