"""Convenience CLI for the fastest live synthetic-data sanity run."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .main import ALL_STEPS, SyntheticDataPipeline

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "smoke" / "smoke_sanity_check.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the smoke sanity config. This is the smallest live data-generation "
            "run and skips TTS/audio."
        ),
    )
    parser.add_argument(
        "--step",
        nargs="+",
        choices=ALL_STEPS,
        help="Which steps to run (default: all)",
    )
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--run-id", help="Override run ID")
    parser.add_argument("--output-root", help="Override output root")
    parser.add_argument("--force", action="store_true", help="Regenerate all outputs")

    args = parser.parse_args()

    pipeline = SyntheticDataPipeline(str(DEFAULT_CONFIG_PATH))
    if args.seed is not None:
        pipeline.settings.seed = args.seed
    if args.run_id:
        pipeline.settings.run_id = args.run_id
    if args.output_root:
        pipeline.settings.output_root = args.output_root
    pipeline.force = args.force

    asyncio.run(pipeline.run(args.step))


if __name__ == "__main__":
    main()
