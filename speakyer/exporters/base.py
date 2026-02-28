from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speakyer.cards.base import Card


class CardExporter(ABC):
    """Push a batch of pending cards to an external flashcard service."""

    @abstractmethod
    def export(self, cards: list["Card"]) -> list[int]:
        """Export *cards* and return the list of external note IDs in the same order."""
