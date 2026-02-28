#!/usr/bin/env python3
"""Run the Speakyer pipeline.

Usage:
    python scripts/run_pipeline.py                        # all active sources
    python scripts/run_pipeline.py --source tagesschau   # one source
    python scripts/run_pipeline.py --dry-run             # preview only
    python scripts/run_pipeline.py --stage download      # up to download stage
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speakyer.database import init
from speakyer.pipeline.base import PipelineContext
from speakyer.pipeline.stages import DownloadStage, TranscribeStage
from speakyer.sources.loader import get_known_guids, insert_episodes, sync_sources
from speakyer.sources.rss import RSSPodcastSource

# Ordered list of available stages. Extended each milestone.
STAGES = {
    "download": DownloadStage,
    "transcribe": TranscribeStage,
    # "nlp": NLPStage,                # Milestone 3
    # "cards": CardGenStage,          # Milestone 4
    # "export": ExportStage,          # Milestone 5
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Speakyer pipeline.")
    parser.add_argument("--source", help="Source name to process (default: all active)")
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

    sources = sync_sources()
    if not sources:
        print("No active sources found. Add entries to sources.yaml.")
        sys.exit(1)

    if args.source:
        sources = [s for s in sources if s.name == args.source]
        if not sources:
            print(f"Source {args.source!r} not found or not active.")
            sys.exit(1)

    stage_names = list(STAGES)
    if args.stage:
        stage_names = stage_names[: stage_names.index(args.stage) + 1]
    stages = [STAGES[n]() for n in stage_names]

    rss = RSSPodcastSource()
    total_new = total_downloaded = 0

    for source in sources:
        print(f"\n[{source.name}] Checking for new episodes...")
        known = get_known_guids(source.id)
        new_episodes = rss.get_new_episodes(source, known)

        if not new_episodes:
            print("  No new episodes.")
            continue

        if args.dry_run:
            # In dry-run mode, never touch the DB — report against the RSS results directly.
            total_new += len(new_episodes)
            episodes_to_process = new_episodes
        else:
            inserted = insert_episodes(new_episodes)
            total_new += len(inserted)
            episodes_to_process = inserted

        print(f"  {len(episodes_to_process)} new episode(s).")

        for ep in episodes_to_process:
            ctx = PipelineContext(source_config=source, episode=ep, dry_run=args.dry_run)
            for stage in stages:
                ctx = stage.run(ctx)
            if ctx.audio_path:
                total_downloaded += 1

    print(f"\nDone — {total_new} new episode(s), {total_downloaded} downloaded.")


if __name__ == "__main__":
    main()
