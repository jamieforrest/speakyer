"""AnkiConnect exporter — pushes pending cards to Anki desktop via HTTP API.

AnkiConnect must be installed in Anki (add-on code 2055492159) and Anki must
be running.  If the connection is refused the exporter raises ``ConnectionError``
with a human-readable message so the pipeline stage can surface it cleanly.
"""

from __future__ import annotations

import requests

from speakyer.cards.base import Card
from speakyer.exporters.base import CardExporter

_DEFAULT_URL = "http://localhost:8765"
_API_VERSION = 6


def _invoke(action: str, url: str, **params) -> object:
    payload = {"action": action, "version": _API_VERSION, "params": params}
    try:
        resp = requests.post(url, json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to AnkiConnect at %s.\n"
            "Make sure Anki is running and the AnkiConnect add-on (2055492159) is installed." % url
        )
    resp.raise_for_status()
    body = resp.json()
    if body.get("error"):
        raise RuntimeError("AnkiConnect error: %s" % body["error"])
    return body["result"]


class AnkiConnectExporter(CardExporter):
    """Export cards to Anki via AnkiConnect.

    Each card row must be enriched with word/episode/source context before
    calling :meth:`export`.  Use :meth:`build_note` to construct the note
    payload from a DB row, or call :meth:`export_rows` directly with raw
    query results.
    """

    def __init__(self, url: str = _DEFAULT_URL) -> None:
        self.url = url

    # ------------------------------------------------------------------
    # CardExporter protocol
    # ------------------------------------------------------------------

    def export(self, cards: list[Card]) -> list[int]:
        """Not used directly — call :meth:`export_rows` from ExportStage."""
        raise NotImplementedError("Use export_rows() which has the full word context.")

    # ------------------------------------------------------------------
    # Main entry point used by ExportStage
    # ------------------------------------------------------------------

    def export_rows(self, rows: list) -> list[int]:
        """Push a batch of enriched card rows to Anki.

        *rows* is a list of sqlite3.Row objects with the columns produced by
        the ExportStage query (card_id, word_id, lemma, surface_form, pos,
        cefr_level, example_sentence, deck_name, episode_title,
        source_name, published_at).

        Returns a list of Anki note IDs in the same order as *rows*.
        """
        self._ensure_deck(rows[0]["deck_name"] if rows else "Speakyer")

        notes = [self._build_note(row) for row in rows]
        note_ids: list = _invoke("addNotes", self.url, notes=notes)  # type: ignore[assignment]

        # addNotes returns null for notes that were rejected (e.g. duplicate).
        # Replace None entries with 0 so callers always get an int list.
        return [nid or 0 for nid in note_ids]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_deck(self, deck_name: str) -> None:
        _invoke("createDeck", self.url, deck=deck_name)

    def _build_note(self, row) -> dict:
        front = self._build_front(row)
        back = self._build_back(row)
        tags = self._build_tags(row)
        return {
            "deckName": row["deck_name"],
            "modelName": "Basic",
            "fields": {"Front": front, "Back": back},
            "tags": tags,
            "options": {"allowDuplicate": False, "duplicateScope": "deck"},
        }

    @staticmethod
    def _build_front(row) -> str:
        sentence = (row["example_sentence"] or "").strip()
        surface = row["surface_form"]
        # Bold the first case-insensitive match of the surface form.
        lower = sentence.lower()
        idx = lower.find(surface.lower())
        if idx != -1:
            sentence = (
                sentence[:idx]
                + "<b>"
                + sentence[idx : idx + len(surface)]
                + "</b>"
                + sentence[idx + len(surface) :]
            )
        return sentence

    @staticmethod
    def _build_back(row) -> str:
        parts = [f"{row['lemma']} · {row['pos']}"]
        if row["cefr_level"]:
            parts.append(f"CEFR {row['cefr_level']}")
        parts.append(row["source_name"])
        if row["episode_title"]:
            parts.append(f"<i>{row['episode_title']}</i>")
        return " · ".join(parts)

    @staticmethod
    def _build_tags(row) -> list[str]:
        tags = ["speakyer"]
        if row["cefr_level"]:
            tags.append(f"cefr::{row['cefr_level']}")
        source = row["source_name"].replace(" ", "_").lower()
        tags.append(f"podcast::{source}")
        return tags
