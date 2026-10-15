"""Build the normalized AMI reference directory used by the realism judge.

The realism judge/refine loop in step 4 shows the judge an excerpt of a real
AMI meeting next to the synthetic candidate. This script renders each AMI
meeting from the manual annotations (``ami_public_manual_1.6.2``, available
from https://groups.inf.ed.ac.uk/ami/download/) as ``Speaker N: ...`` lines
and writes the deterministic style-reference / discriminator-eval split.

Usage::

    python scripts/prepare_ami_reference.py \
        --ami-root data/ami/ami_public_manual_1.6.2 \
        --output-dir data/ami/normalized

Then set ``generation.realism_rewrite.ami_reference_dir`` to the output dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from osynth.tts.ami_normalize import list_ami_meetings, normalize_ami_meeting, write_ami_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ami-root", type=Path, required=True,
                        help="Directory containing the AMI manual annotations (words/, segments/)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meeting_ids = []
    for meeting_id in list_ami_meetings(args.ami_root):
        text = normalize_ami_meeting(args.ami_root, meeting_id)
        if not text.strip():
            continue
        (args.output_dir / f"{meeting_id}.txt").write_text(text, encoding="utf-8")
        meeting_ids.append(meeting_id)
    style, evaluation = write_ami_split(args.output_dir, meeting_ids=meeting_ids, overwrite=args.overwrite)
    print(f"Wrote {len(meeting_ids)} meetings to {args.output_dir}; split files: {style}, {evaluation}")


if __name__ == "__main__":
    main()
