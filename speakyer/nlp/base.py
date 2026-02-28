from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Word:
    """A vocabulary word extracted from a transcript segment."""

    episode_id: int
    surface_form: str
    lemma: str
    pos: str
    cefr_level: str | None
    example_sentence: str
    start_time: float | None
    transcript_segment_id: int | None = None


class NLPExtractor(ABC):
    """Abstract interface for NLP extraction.

    Receives one segment at a time together with a pre-loaded CEFR lookup
    dictionary and returns extracted vocabulary words.
    """

    @abstractmethod
    def extract(
        self,
        text: str,
        episode_id: int,
        segment_id: int,
        start_time: float,
        cefr_lookup: dict[str, str],
    ) -> list[Word]: ...
