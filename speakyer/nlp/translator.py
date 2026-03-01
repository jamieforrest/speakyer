"""Sentence translator using a local HuggingFace model.

Translates German example sentences to English for Anki card backs.
The model is downloaded from HuggingFace Hub on first use and cached
in the default HuggingFace cache directory (~/.cache/huggingface).

Recommended models (choose via SPEAKYER_TRANSLATION_MODEL env var):
  - Helsinki-NLP/opus-mt-de-en        ~310 MB, fast, good quality  (default)
  - Helsinki-NLP/opus-mt-tc-big-de-en ~1.2 GB, better quality
  - facebook/nllb-200-distilled-600M  ~2.4 GB, multilingual

Usage::

    from speakyer.nlp.translator import Translator
    t = Translator()
    print(t.translate("Die Bundesregierung hat beschlossen, dass die Steuern erhöht werden."))
    # → "The federal government has decided to raise taxes."
"""

from __future__ import annotations


try:
    import transformers as _transformers  # noqa: F401

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_DEFAULT_MODEL = "Helsinki-NLP/opus-mt-de-en"
_BATCH_SIZE = 32  # sentences per forward pass


class Translator:
    """Translate sentences using a local HuggingFace seq2seq model.

    Uses ``AutoTokenizer`` + ``AutoModelForSeq2SeqLM`` directly instead of
    the ``pipeline`` helper, which avoids the ``translation_XX_to_YY`` task
    naming requirement introduced in recent ``transformers`` versions.

    The model is loaded lazily on first call to :meth:`translate` or
    :meth:`translate_batch`.

    Args:
        model: HuggingFace model ID. Defaults to ``Helsinki-NLP/opus-mt-de-en``.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. "
                "Run: pip install transformers torch"
            )
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"  Loading translation model {self.model!r} (first use — may download) ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model)

    def translate(self, text: str) -> str:
        """Translate *text* and return the translated string.

        Returns an empty string for blank input without loading the model.
        """
        if not text or not text.strip():
            return ""
        results = self.translate_batch([text])
        return results[0]

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a list of texts, returning a same-length list of strings.

        Empty strings are passed through without loading the model.
        Internally processes in chunks of ``_BATCH_SIZE`` to bound memory use.
        """
        if not texts:
            return []

        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty:
            return [""] * len(texts)

        self._load()

        translations: list[str] = [""] * len(texts)

        for chunk_start in range(0, len(non_empty), _BATCH_SIZE):
            chunk = non_empty[chunk_start : chunk_start + _BATCH_SIZE]
            indices, batch_texts = zip(*chunk)

            inputs = self._tokenizer(
                list(batch_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = self._model.generate(**inputs, max_new_tokens=512)
            decoded = [
                self._tokenizer.decode(o, skip_special_tokens=True)
                for o in outputs
            ]
            for idx, dec in zip(indices, decoded):
                translations[idx] = dec

        return translations
