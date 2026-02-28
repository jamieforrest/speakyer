from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SourceConfig:
    id: int
    name: str
    rss_url: str
    language: str
    transcript_strategy: str
    active: bool


@dataclass
class Episode:
    guid: str
    source_id: int
    title: str | None = None
    published_at: datetime | None = None
    audio_url: str | None = None
    audio_path: str | None = None
    status: str = "fetched"
    id: int | None = None


class ContentSource(ABC):
    @abstractmethod
    def get_new_episodes(
        self, source: SourceConfig, known_guids: set[str]
    ) -> list[Episode]: ...
