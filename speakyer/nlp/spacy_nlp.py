"""spaCy-based NLP extractor.

Loads a spaCy model lazily on first use so that importing the module does not
require the model to be present (enabling imports in test environments where
the model may not be installed).
"""

from __future__ import annotations

from speakyer.nlp.base import NLPExtractor, Word

# POS tags whose tokens are considered learnable vocabulary.
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


try:
    import spacy as _spacy  # noqa: F401

    _SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPACY_AVAILABLE = False


class SpacyExtractor(NLPExtractor):
    """Extract vocabulary words using spaCy.

    Args:
        model: spaCy model name, e.g. ``"de_core_news_lg"``.
    """

    def __init__(self, model: str = "de_core_news_lg") -> None:
        self.model = model
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            if not _SPACY_AVAILABLE:
                raise ImportError(
                    "spacy is not installed. Run: pip install spacy"
                )
            import spacy

            try:
                self._nlp = spacy.load(self.model)
            except OSError as exc:
                raise OSError(
                    f"spaCy model '{self.model}' not found. "
                    f"Run: python -m spacy download {self.model}"
                ) from exc
        return self._nlp

    def extract(
        self,
        text: str,
        episode_id: int,
        segment_id: int,
        start_time: float,
        cefr_lookup: dict[str, str],
    ) -> list[Word]:
        doc = self.nlp(text)
        words: list[Word] = []
        for token in doc:
            if token.pos_ not in CONTENT_POS:
                continue
            if len(token.text.strip()) < 2:
                continue
            lemma = token.lemma_.lower()
            words.append(
                Word(
                    episode_id=episode_id,
                    transcript_segment_id=segment_id,
                    surface_form=token.text,
                    lemma=lemma,
                    pos=token.pos_,
                    cefr_level=cefr_lookup.get(lemma),
                    example_sentence=text,
                    start_time=start_time,
                )
            )
        return words
