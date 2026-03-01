"""Sentence translator using a local HuggingFace model.

Translates German example sentences to English for Anki card backs.
The model is downloaded from HuggingFace Hub on first use and cached
in the default HuggingFace cache directory (~/.cache/huggingface).

Recommended models (choose via SPEAKYER_TRANSLATION_MODEL env var):
  - Helsinki-NLP/opus-mt-de-en   ~310 MB, fast, good quality  (default)
  - Helsinki-NLP/opus-mt-tc-big-de-en  ~1.2 GB, better quality
  - facebook/nllb-200-distilled-600M  ~2.4 GB, multilingual

Usage::

    from speakyer.nlp.translator import Translator
    t = Translator()
    print(t.translate("Die Bundesregierung hat beschlossen, dass die Steuern erhöht werden."))
    # → "The federal government has decided to raise taxes."
"""

from __future__ import annotations


try:
    from transformers import pipeline as _hf_pipeline  # noqa: F401

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_DEFAULT_MODEL = "Helsinki-NLP/opus-mt-de-en"


class Translator:
    """Translate sentences using a local HuggingFace MarianMT model.

    The model is loaded lazily on first call to :meth:`translate`.

    Args:
        model: HuggingFace model ID. Defaults to ``Helsinki-NLP/opus-mt-de-en``.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers is not installed. "
                    "Run: pip install transformers torch"
                )
            from transformers import pipeline

            print(f"  Loading translation model {self.model!r} (first use — may download) ...")
            self._pipeline = pipeline("translation", model=self.model)
        return self._pipeline

    def translate(self, text: str) -> str:
        """Translate *text* and return the translated string.

        Returns an empty string for blank input without loading the model.
        """
        if not text or not text.strip():
            return ""
        result = self.pipeline(text.strip(), max_length=512)
        return result[0]["translation_text"]

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a list of texts in one batch call.

        Empty strings are passed through without translation.  Returns a
        list of the same length as *texts*.
        """
        if not texts:
            return []

        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty:
            return [""] * len(texts)

        indices, to_translate = zip(*non_empty)
        results = self.pipeline(list(to_translate), max_length=512)
        translations = [""] * len(texts)
        for idx, res in zip(indices, results):
            translations[idx] = res["translation_text"]
        return translations
