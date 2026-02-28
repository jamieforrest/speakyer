import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    db_path: Path
    data_dir: Path
    sources_yaml: Path
    whisper_model: str
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
            anki_connect_url=os.getenv(
                "SPEAKYER_ANKI_CONNECT_URL", "http://localhost:8765"
            ),
        )


config = Config.from_env()
