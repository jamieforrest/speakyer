#!/usr/bin/env python3
"""Run the Speakyer pipeline.

Usage:
    python scripts/run_pipeline.py                        # all active sources
    python scripts/run_pipeline.py --source tagesschau   # one source
    python scripts/run_pipeline.py --episode-id 1        # single episode by DB id
    python scripts/run_pipeline.py --dry-run             # preview only
    python scripts/run_pipeline.py --stage download      # up to download stage
    python scripts/run_pipeline.py --stage transcribe    # up to transcribe stage
    python scripts/run_pipeline.py --stage nlp           # up to NLP/CEFR tagging
    python scripts/run_pipeline.py --stage cards         # up to card generation
    python scripts/run_pipeline.py --stage audio-clip    # up to audio clip extraction
    python scripts/run_pipeline.py --stage export        # up to Anki export (requires Anki running)

    # Add audio clips to existing Anki cards (requires Anki running):
    python scripts/run_pipeline.py --pipeline audio-update
    python scripts/run_pipeline.py --pipeline audio-update --episode-id 1
    python scripts/run_pipeline.py --pipeline audio-update --source tagesschau

    # Override the Whisper model for this run (faster models for development):
    python scripts/run_pipeline.py --whisper-model mlx-community/whisper-small-mlx
    python scripts/run_pipeline.py --whisper-model mlx-community/whisper-medium-mlx

    Available mlx-whisper models (fastest → most accurate):
      mlx-community/whisper-tiny-mlx        ~32x realtime  (rough accuracy)
      mlx-community/whisper-base-mlx        ~16x realtime
      mlx-community/whisper-small-mlx       ~6x  realtime  (good for dev)
      mlx-community/whisper-medium-mlx      ~2x  realtime  (good balance)
      mlx-community/whisper-large-v3-mlx    ~1x  realtime  (default, best)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import speakyer.config as _cfg_module
from speakyer.database import init
from speakyer.pipeline.base import PipelineContext
from speakyer.pipeline.stages import (
    AudioClipStage,
    CardGenStage,
    CardUpdateStage,
    DownloadStage,
    ExportStage,
    NLPStage,
    TranscribeStage,
    AUDIO_UPDATE_PIPELINE,
)
from speakyer.sources.loader import (
    get_episode_by_id,
    get_known_guids,
    get_pending_episodes,
    insert_episodes,
    sync_sources,
)
from speakyer.sources.rss import RSSPodcastSource

# Ordered list of available stages. Extended each milestone.
STAGES = {
    "download": DownloadStage,
    "transcribe": TranscribeStage,
    "nlp": NLPStage,
    "cards": CardGenStage,
    "audio-clip": AudioClipStage,
    "export": ExportStage,
}

# Statuses that indicate an episode needs processing, per stage.
# An episode at any of these statuses will be picked up when that stage is included.
STAGE_PENDING_STATUSES = {
    "download": {"fetched"},
    "transcribe": {"fetched", "downloaded"},
    "nlp": {"fetched", "downloaded", "transcribed"},
    "cards": {"fetched", "downloaded", "transcribed", "analyzed"},
    "audio-clip": {"fetched", "downloaded", "transcribed", "analyzed", "cards_pending"},
    "export": {"fetched", "downloaded", "transcribed", "analyzed", "cards_pending"},
}


def _run_audio_update_pipeline(args) -> None:
    """Run AudioClipStage + CardUpdateStage across episodes with exported cards."""
    from speakyer.database import db
    from speakyer.sources.base import Episode, SourceConfig

    with db() as conn:
        ep_rows = conn.execute(
            """
            SELECT DISTINCT e.id, e.guid, e.title, e.audio_path, e.status,
                            e.source_id,
                            s.name AS source_name, s.rss_url,
                            s.language, s.transcript_strategy, s.active
            FROM episodes e
            JOIN words w ON w.episode_id = e.id
            JOIN cards c ON c.word_id = w.id
            JOIN sources s ON s.id = e.source_id
            WHERE c.status IN ('pending', 'exported')
            ORDER BY e.id
            """
        ).fetchall()

    if args.episode_id:
        ep_rows = [r for r in ep_rows if r["id"] == args.episode_id]
    if args.source:
        ep_rows = [r for r in ep_rows if r["source_name"] == args.source]

    if not ep_rows:
        print("No episodes with cards found.")
        return

    stages = [cls() for cls in AUDIO_UPDATE_PIPELINE]
    print(f"Running audio-update pipeline for {len(ep_rows)} episode(s)...")

    for row in ep_rows:
        ep = Episode(
            id=row["id"],
            guid=row["guid"],
            source_id=row["source_id"],
            title=row["title"],
            audio_path=row["audio_path"],
            status=row["status"],
        )
        source = SourceConfig(
            id=row["source_id"],
            name=row["source_name"],
            rss_url=row["rss_url"],
            language=row["language"],
            transcript_strategy=row["transcript_strategy"],
            active=bool(row["active"]),
        )
        print(f"\n[{source.name}] {ep.title!r}")
        ctx = PipelineContext(source_config=source, episode=ep, dry_run=args.dry_run)
        for stage in stages:
            ctx = stage.run(ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Speakyer pipeline.")
    parser.add_argument("--source", help="Source name to process (default: all active)")
    parser.add_argument("--episode-id", type=int, help="Process a single episode by DB id")
    parser.add_argument(
        "--stage",
        choices=list(STAGES),
        help="Run up to and including this stage (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing anything"
    )
    parser.add_argument(
        "--whisper-model",
        metavar="MODEL",
        help="Override the Whisper model for this run (e.g. mlx-community/whisper-small-mlx)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["audio-update"],
        help="Run a named auxiliary pipeline instead of the main one",
    )
    args = parser.parse_args()

    if args.whisper_model:
        _cfg_module.config.whisper_model = args.whisper_model
        print(f"Using Whisper model: {args.whisper_model}")

    init()

    if args.pipeline == "audio-update":
        _run_audio_update_pipeline(args)
        return

    stage_names = list(STAGES)
    if args.stage:
        stage_names = stage_names[: stage_names.index(args.stage) + 1]
    stages = [STAGES[n]() for n in stage_names]

    # Statuses that need processing given the selected stage set.
    pending_statuses: set[str] = set()
    for n in stage_names:
        pending_statuses |= STAGE_PENDING_STATUSES[n]

    # --- Single episode mode ---
    if args.episode_id:
        result = get_episode_by_id(args.episode_id)
        if not result:
            print(f"Episode {args.episode_id} not found.")
            sys.exit(1)
        ep, source = result
        print(f"\n[{source.name}] Processing episode {ep.id}: {ep.title!r}")
        ctx = PipelineContext(source_config=source, episode=ep, dry_run=args.dry_run)
        for stage in stages:
            ctx = stage.run(ctx)
        downloaded = 1 if ctx.audio_path else 0
        print(f"\nDone — 1 episode, {downloaded} downloaded.")
        return

    # --- Source-based mode ---
    sources = sync_sources()
    if not sources:
        print("No active sources found. Add entries to sources.yaml.")
        sys.exit(1)

    if args.source:
        sources = [s for s in sources if s.name == args.source]
        if not sources:
            print(f"Source {args.source!r} not found or not active.")
            sys.exit(1)

    rss = RSSPodcastSource()
    total_new = total_downloaded = 0

    for source in sources:
        print(f"\n[{source.name}] Checking for new episodes...")
        known = get_known_guids(source.id)
        new_episodes = rss.get_new_episodes(source, known)

        if args.dry_run:
            total_new += len(new_episodes)
            episodes_to_process = new_episodes
        else:
            inserted = insert_episodes(new_episodes)
            total_new += len(inserted)
            # Load all pending episodes for this source (includes newly inserted ones).
            episodes_to_process = get_pending_episodes(source.id, pending_statuses)

        if not episodes_to_process:
            print("  No episodes to process.")
            continue

        print(f"  {len(episodes_to_process)} episode(s) to process.")

        for ep in episodes_to_process:
            ctx = PipelineContext(source_config=source, episode=ep, dry_run=args.dry_run)
            for stage in stages:
                ctx = stage.run(ctx)
            if ctx.audio_path:
                total_downloaded += 1

    print(f"\nDone — {total_new} new episode(s), {total_downloaded} downloaded.")


if __name__ == "__main__":
    main()
