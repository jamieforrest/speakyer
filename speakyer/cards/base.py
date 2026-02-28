from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Card:
    """A pending Anki card derived from a single word occurrence.

    ``word_id`` links back to the ``words`` table row that was chosen as the
    canonical example for this lemma.  The actual Anki note content (front /
    back HTML, tags) is derived from the joined word + episode data at export
    time (Milestone 5).
    """

    word_id: int
    deck_name: str
    status: str = "pending"
    id: int | None = None
