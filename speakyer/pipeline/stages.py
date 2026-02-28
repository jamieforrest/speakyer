import requests

from speakyer.config import config
from speakyer.database import db
from speakyer.pipeline.base import PipelineContext, PipelineStage
from speakyer.storage import LocalStorage


class DownloadStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> PipelineContext:
        ep = ctx.episode

        if not ep.audio_url:
            print(f"  [skip] {ep.title!r}: no audio URL")
            return ctx

        storage = LocalStorage(config.data_dir)
        rel_path = f"audio/{ep.id}.mp3"

        if storage.exists(rel_path):
            print(f"  [skip] {ep.title!r}: already downloaded")
            ctx.audio_path = storage.absolute_path(rel_path)
            return ctx

        if ctx.dry_run:
            print(f"  [dry-run] would download: {ep.title!r}")
            return ctx

        print(f"  Downloading: {ep.title!r} ...")
        with requests.get(ep.audio_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            audio_data = b"".join(response.iter_content(chunk_size=65536))

        storage.write(rel_path, audio_data)
        ctx.audio_path = storage.absolute_path(rel_path)

        with db() as conn:
            conn.execute(
                "UPDATE episodes SET audio_path = ?, status = ? WHERE id = ?",
                (rel_path, "downloaded", ep.id),
            )

        size_mb = len(audio_data) / 1_048_576
        print(f"  Saved {size_mb:.1f} MB → {ctx.audio_path}")
        return ctx
