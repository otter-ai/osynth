"""Step 10: Room simulation — apply conference-room acoustics to in-room speakers.

Consumes per-speaker WAVs from ``step_9_audio_finalize`` and produces, per
meeting with at least one ``IN_ROOM`` participant:

- ``combined_room.wav`` — microphone-perspective recording of the in-room
  speakers (RIR-convolved + background / AC / event / white-noise layers).
- per-speaker room-convolved WAVs for each in-room speaker.
- ``combined_room_full.wav`` — the room mic recording with each
  remote/phone participant's dry per-speaker track from step 9 summed in,
  so a single file represents what would be captured by the conference
  room mic plus the call dial-in.

This step is opt-in via ``settings.room_simulation.enabled`` so existing
configs are unaffected unless they turn it on.

The step is idempotent per-meeting: re-running without ``--force`` only
regenerates whichever of ``combined_room.wav`` / ``combined_room_full.wav``
are missing, leaving existing outputs (and the expensive RIR convolution)
untouched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..models.calendar import MeetingJoinSpec
from ..models.settings import SyntheticDataSettings
from ..tts.room_simulation import (
    in_room_speaker_ids,
    mix_room_with_remote_tracks,
    remote_speaker_ids,
    resolve_noise_paths,
    simulate_room,
)
from ..utils import fileio

logger = logging.getLogger(__name__)


COMBINED_ROOM_FILENAME = "combined_room.wav"
COMBINED_ROOM_FULL_FILENAME = "combined_room_full.wav"


class Step10RoomSimulation:
    """Apply room acoustics and background noise to meetings with in-room speakers."""

    def __init__(
        self,
        settings: SyntheticDataSettings,
        output_root: str,
        data: dict,
    ) -> None:
        self.settings = settings
        self.data = data
        self.run_dir = Path(output_root)
        self.finalize_dir = self.run_dir / "step_9_audio_finalize"
        self.output_dir = self.run_dir / "step_10_room_simulation"
        self.generate_audio: bool = settings.generation.generate_audio
        self.room_cfg = settings.room_simulation
        self.artifacts: dict[str, dict] = data.get("artifacts", {})
        self.seed: int = settings.seed or 42

    async def run(self, force: bool = False) -> dict:
        if not self.generate_audio:
            logger.info("Audio generation disabled, skipping room simulation.")
            return {}
        if not self.room_cfg.enabled:
            logger.info("Room simulation disabled in settings, skipping.")
            return {}

        fileio.makedirs(self.output_dir)

        results_path = self.output_dir / "room_results.json"
        prior_results = self._read_results_json(results_path)
        results: dict[str, dict] = {}

        meeting_ids = sorted(
            d.name for d in self.finalize_dir.iterdir() if d.is_dir()
        ) if self.finalize_dir.exists() else []

        noise_paths = None  # resolved lazily on first cache miss
        skipped = 0
        ran_simulate = 0
        ran_mix = 0
        cached = 0

        for meeting_id in meeting_ids:
            artifact = self.artifacts.get(meeting_id)
            if artifact is None or artifact.get("type") != "meeting":
                continue

            raw_spec = artifact.get("join_spec")
            if not raw_spec:
                skipped += 1
                continue
            try:
                join_spec = MeetingJoinSpec.model_validate(raw_spec)
            except Exception:
                logger.exception(
                    "Invalid join_spec for %s; skipping room simulation", meeting_id,
                )
                skipped += 1
                continue

            in_room_ids = in_room_speaker_ids(join_spec)
            if not in_room_ids:
                skipped += 1
                continue

            speaker_audio_paths = artifact.get("speaker_audio_paths") or {}
            in_room_wavs = {
                pid: Path(speaker_audio_paths[pid])
                for pid in in_room_ids
                if pid in speaker_audio_paths
                and Path(speaker_audio_paths[pid]).exists()
            }
            if not in_room_wavs:
                logger.warning(
                    "No in-room speaker WAVs available for %s; skipping", meeting_id,
                )
                skipped += 1
                continue

            meeting_out_dir = self.output_dir / meeting_id
            combined_room_path = meeting_out_dir / COMBINED_ROOM_FILENAME
            combined_full_path = meeting_out_dir / COMBINED_ROOM_FULL_FILENAME

            prior_meeting = prior_results.get(meeting_id, {})

            # ----- Phase A: RIR convolution (expensive, reuse when possible).
            need_simulate = force or not combined_room_path.exists()
            if need_simulate:
                if noise_paths is None:
                    noise_paths = resolve_noise_paths(
                        self.room_cfg, self.settings.output_root,
                    )
                    logger.info(
                        "Room simulation noise paths: background=%s hum=%s events=%s",
                        noise_paths.background_noise_dir,
                        noise_paths.hum_noise_dir,
                        noise_paths.event_noise_dir,
                    )
                try:
                    sim_result = simulate_room(
                        in_room_speaker_wavs=in_room_wavs,
                        timing_data=artifact.get("timing_data") or {},
                        join_spec=join_spec,
                        noise_paths=noise_paths,
                        room_cfg=self.room_cfg,
                        out_dir=meeting_out_dir,
                        seed=self.seed,
                        meeting_id=meeting_id,
                    )
                except Exception:
                    logger.exception("Room simulation failed for %s", meeting_id)
                    continue
                speaker_room_paths = {
                    k: str(v) for k, v in sim_result.speaker_room_paths.items()
                }
                rt60_measured = sim_result.rt60_measured
                noise_config = sim_result.noise_config
                ran_simulate += 1
            else:
                speaker_room_paths = dict(prior_meeting.get("speaker_room_paths", {}))
                rt60_measured = prior_meeting.get("rt60_measured", 0.0)
                noise_config = prior_meeting.get("noise_config", {}) or {}

            # ----- Phase B: room + remote dial-in mix (cheap, always
            # produced; reused when both files already exist and ``force``
            # is false).
            remote_ids = remote_speaker_ids(join_spec)
            remote_wavs: dict[str, Path] = {
                pid: Path(speaker_audio_paths[pid])
                for pid in remote_ids
                if pid in speaker_audio_paths
                and Path(speaker_audio_paths[pid]).exists()
            }
            missing_remote = sorted(remote_ids - set(remote_wavs))
            if missing_remote:
                logger.warning(
                    "Missing dry per-speaker WAVs for remote participants in %s: %s",
                    meeting_id, missing_remote,
                )

            need_mix = force or not combined_full_path.exists()
            if need_mix:
                try:
                    mix_room_with_remote_tracks(
                        combined_room_path=combined_room_path,
                        remote_speaker_wavs=[remote_wavs[pid] for pid in sorted(remote_wavs)],
                        out_path=combined_full_path,
                        sample_rate=int(self.room_cfg.sample_rate),
                    )
                except Exception:
                    logger.exception(
                        "Final room+remote mix failed for %s", meeting_id,
                    )
                    continue
                ran_mix += 1
            elif not need_simulate:
                cached += 1

            results[meeting_id] = {
                "combined_room_path": str(combined_room_path),
                "combined_room_full_path": str(combined_full_path),
                "speaker_room_paths": speaker_room_paths,
                "remote_speaker_paths": {
                    pid: str(p) for pid, p in remote_wavs.items()
                },
                "rt60_measured": rt60_measured,
                "noise_config": noise_config,
            }

            artifact["room_audio_path"] = str(combined_room_path)
            artifact["combined_room_full_path"] = str(combined_full_path)
            artifact["speaker_room_paths"] = speaker_room_paths
            artifact["remote_speaker_paths"] = results[meeting_id]["remote_speaker_paths"]

        if results:
            fileio.write_json(results_path, results)

        logger.info(
            "Step 10 room simulation complete: %d meetings (simulated=%d, mixed=%d, "
            "fully cached=%d, skipped=%d)",
            len(results), ran_simulate, ran_mix, cached, skipped,
        )
        return {"room_results": results}

    def _read_results_json(self, results_path: Path) -> dict[str, dict]:
        if not results_path.exists():
            return {}
        try:
            return json.loads(results_path.read_text())
        except Exception:
            logger.exception(
                "Failed to read existing %s; treating as empty", results_path,
            )
            return {}

    def _load_from_disk(self) -> dict:
        results_path = self.output_dir / "room_results.json"
        if not results_path.exists():
            return {"room_results": {}}
        results = json.loads(results_path.read_text())
        for meeting_id, info in results.items():
            artifact = self.artifacts.get(meeting_id)
            if artifact is not None:
                artifact["room_audio_path"] = info.get("combined_room_path")
                artifact["combined_room_full_path"] = info.get("combined_room_full_path")
                artifact["speaker_room_paths"] = info.get("speaker_room_paths", {})
                artifact["remote_speaker_paths"] = info.get("remote_speaker_paths", {})
        return {"room_results": results}
