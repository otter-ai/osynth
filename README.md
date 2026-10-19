# OtterSynth

**OtterSynth** is a synthetic meeting corpus. This repository contains the generation pipeline (code and prompts) for the paper *OSynth: A Longitudinal, Context-Sensitive Evaluation of Meeting Understanding* (EMNLP 2026).

## Sample

https://github.com/user-attachments/assets/8780e2af-19c9-4ee3-b6af-dabc22883a85

## Running

(Before running perform the setup below.)

```bash
# Small smoke test
uv run osynth-smoke

# Full two-week workspace with audio
uv run osynth --config osynth/config/example_14d.yaml
```

## Data

The full corpus release is pending. We will udpate here once it is available.

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).
Run the commands below from the repository root. Preparing the voice cache
also requires `ffmpeg` on your `PATH`.

```bash
uv sync                         # text generation (steps 1-7, 11)
uv sync --extra tts              # + step 8 TTS (CUDA GPU)
uv sync --extra tts --extra nfa  # full audio example, including forced alignment
```

API keys go in `.env`.

```
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

External inputs for the two-week example (the smoke run needs none):

| Input | Used by | How to prepare |
|---|---|---|
| Voice cache (`data/voice_cache`) | steps 1, 7, 8 | `uv run python scripts/prepare_voice_cache.py --corpus-dir <Common Voice SPS dir> --output-dir data/voice_cache` |
| AMI reference (`data/ami/normalized`) | step 4 realism judge | `uv run python scripts/prepare_ami_reference.py --ami-root <ami_public_manual_1.6.2> --output-dir data/ami/normalized` |
| NeMo Forced Aligner (`third_party/NeMo`) | step 9 | `git clone --depth 1 --branch v3.0.0 https://github.com/NVIDIA/NeMo.git third_party/NeMo` (match the installed `nemo-toolkit` version) |
| Noise audio (`data/noise/`) | step 10 | Supply local WAV files and event metadata as described below. |

Step 9 runs NeMo's forced aligner (`tools/nemo_forced_aligner/align.py`),
which is not bundled here; its path is `alignment.nfa_script`. Setting
`alignment.method: energy` skips it and uses energy-based speech boundaries.

Voice-cache, AMI, and aligner paths are relative to the working directory. Noise paths
in the YAML are relative to `output_root`, so the example uses
`../data/noise/...` with `output_root: outputs`. Prompt paths are relative to
the installed `osynth` package. Absolute paths work for these inputs as well.

### Local noise files

Supply your own nonempty WAV recordings and configure their directories under
`room_simulation`. Use mono, 16 kHz PCM. Other formats are converted on load,
but resampling is a simple linear interpolation without anti-aliasing, so
resample to 16 kHz beforehand (e.g. `sox in.wav -r 16000 -c 1 out.wav`).

- **Background** (`background_noise_dir`):
  `<directory>/<environment>/*.wav`, with environments listed in
  `background_environments`.
- **Continuous hum** (`hum_noise_dir`): `<directory>/ac_unit/*.wav`.
- **Short events** (`event_noise_dir`): `<directory>/audio/*.wav`, with a
  `<directory>/meta/events.csv` index containing `filename,category` columns.
  Filenames are relative to `audio/`; categories must match
  `event_categories`.

Directory paths are relative to `output_root`, or may be absolute. Set a path
to `""` to omit that noise layer. Nonempty paths must exist.

## License

- **Code:** [Apache License 2.0](LICENSE) (see [NOTICE](NOTICE)). It is
  provided "as is", without warranties or conditions of any kind; see
  sections 7–8 of the license.
- **Data:** [CC BY-NC 4.0](LICENSE-DATA.md). Upstream conditions (voices,
  model outputs, references) are listed there.

## Citation

```bibtex
@inproceedings{porada-etal-2026-osynth,
    title = "{OS}ynth: A Longitudinal, Context-Sensitive Evaluation of Meeting Understanding",
    author = "Porada, Ian and
      Tripathi, Suraj and
      Animesh, Chaitanya and
      Sari, Leda and
      Narisetty, Chaitanya and
      Matthews, Austin R. and
      Yu, Bohan and
      Thulasiram, Ramesh L. and
      Sell, Gregory and
      Chin, Kean and
      Genzel, Dmitriy",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    year = "2026",
    publisher = "Association for Computational Linguistics",
}
```
