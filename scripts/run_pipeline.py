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
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speakyer.database import init
from speakyer.pipeline.base import PipelineContext
from speakyer.pipeline.stages import DownloadStage, NLPStage, TranscribeStage
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
    # "cards": CardGenStage,          # Milestone 4
    # "export": ExportStage,          # Milestone 5
}

# Statuses that indicate an episode needs processing, per stage.
# An episode at any of these statuses will be picked up when that stage is included.
STAGE_PENDING_STATUSES = {
    "download": {"fetched"},
    "transcribe": {"fetched", "downloaded"},
    "nlp": {"fetched", "downloaded", "transcribed"},
}


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
    args = parser.parse_args()

    init()

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
