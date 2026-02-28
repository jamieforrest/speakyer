"""Shared fixtures for all tests."""

from pathlib import Path

import pytest

import speakyer.config as _cfg_module
from speakyer.database import init
from speakyer.storage import LocalStorage


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Initialised SQLite database in a temp directory."""
    db_path = tmp_path / "test.db"
    init(db_path)
    return db_path


@pytest.fixture
def tmp_storage(tmp_path: Path) -> LocalStorage:
    """LocalStorage backed by a temp directory."""
    return LocalStorage(tmp_path / "data")


@pytest.fixture
def patched_config(tmp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the global config singleton to temp paths.

    speakyer.database and speakyer.sources.loader both hold a reference to
    the same Config instance (imported via `from speakyer.config import config`).
    Patching the object's attributes here affects all modules simultaneously.
    """
    monkeypatch.setattr(_cfg_module.config, "db_path", tmp_db)
    monkeypatch.setattr(_cfg_module.config, "sources_yaml", tmp_path / "sources.yaml")
    monkeypatch.setattr(_cfg_module.config, "data_dir", tmp_path / "data")
    return _cfg_module.config
