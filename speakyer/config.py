import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def _load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Load KEY=value pairs from a .env file into os.environ (if not already set)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")  # handle simple quoting
        os.environ.setdefault(key, value)


_load_dotenv()


@dataclass
class Config:
    db_path: Path
    data_dir: Path
    sources_yaml: Path
    whisper_model: str
    spacy_model: str
    anki_connect_url: str

    @classmethod
    def from_env(cls) -> "Config":
        base = Path(os.getenv("SPEAKYER_BASE_DIR", str(BASE_DIR)))
        return cls(
            db_path=Path(os.getenv("SPEAKYER_DB_PATH", str(base / "speakyer.db"))),
            data_dir=Path(os.getenv("SPEAKYER_DATA_DIR", str(base / "data"))),
            sources_yaml=Path(
                os.getenv("SPEAKYER_SOURCES_YAML", str(base / "sources.yaml"))
            ),
            whisper_model=os.getenv(
                "SPEAKYER_WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx"
            ),
            spacy_model=os.getenv("SPEAKYER_SPACY_MODEL", "de_core_news_lg"),
            anki_connect_url=os.getenv(
                "SPEAKYER_ANKI_CONNECT_URL", "http://localhost:8765"
            ),
        )


config = Config.from_env()
