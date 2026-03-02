from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    @abstractmethod
    def write(self, relative_path: str, data: bytes) -> None: ...

    @abstractmethod
    def read(self, relative_path: str) -> bytes: ...

    @abstractmethod
    def exists(self, relative_path: str) -> bool: ...

    @abstractmethod
    def absolute_path(self, relative_path: str) -> Path:
        """Return the filesystem path for this file.

        Only meaningful for local storage backends. For remote backends
        (e.g. S3Storage), this should raise NotImplementedError.
        """
        ...


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, relative_path: str, data: bytes) -> None:
        full_path = self.base_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)

    def read(self, relative_path: str) -> bytes:
        return (self.base_dir / relative_path).read_bytes()

    def exists(self, relative_path: str) -> bool:
        return (self.base_dir / relative_path).exists()

    def absolute_path(self, relative_path: str) -> Path:
        return self.base_dir / relative_path
