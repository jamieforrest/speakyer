import click

from speakyer import __version__


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Speakyer: podcast vocabulary learning pipeline."""


@main.command()
@click.option("--source", default=None, help="Source name to process (default: all active)")
@click.option("--dry-run", is_flag=True, help="Show what would be done without writing")
@click.option("--episode-id", type=int, default=None, help="Process a specific episode by ID")
def run(source: str | None, dry_run: bool, episode_id: int | None) -> None:
    """Fetch, transcribe, and export vocabulary from podcasts."""
    click.echo("Pipeline not yet implemented — coming in Milestone 5.")


if __name__ == "__main__":
    main()
