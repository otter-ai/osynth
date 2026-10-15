"""OtterSynth pipeline — main entrypoint.

Generates synthetic workspace data (meetings, emails, documents, slides)
for a simulated company over a configurable time period.

Usage:
    python -m osynth.main [--step STEP ...] [--config PATH]
                                  [--seed INT] [--max-speakers INT] [--force]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .models.settings import (
    SyntheticDataSettings,
    default_synthetic_config_path,
    load_synthetic_data_settings,
)
from .utils import fileio

# Load .env from the working directory (or the nearest parent that has one)
load_dotenv(find_dotenv(usecwd=True))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_STEPS = [
    "step_1",
    "step_2",
    "step_3",
    "step_4",
    "step_5",
    "step_6",
    "step_7",
    "step_8",
    "step_9",
    "step_10",
    "step_11",
]


class SyntheticDataPipeline:
    """Orchestrates the full synthetic data generation pipeline."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or default_synthetic_config_path()
        self.settings: SyntheticDataSettings = load_synthetic_data_settings(self.config_path)
        self.force = False
        self.force_steps: set[str] = set()
        self._data: dict = {}

    def _should_force(self, step_name: str) -> bool:
        return self.force or step_name in self.force_steps

    def _clean_step_output(self, step_name: str, subdirs: list[str]) -> None:
        """Remove stale output files/dirs for a step before regeneration."""
        base = self._get_output_dir()
        for subdir in subdirs:
            path = os.path.join(base, subdir)
            if os.path.exists(path):
                shutil.rmtree(path)
                logger.info("Cleaned stale output: %s", path)

    def _get_output_dir(self) -> str:
        return os.path.join(self.settings.output_root, self.settings.group, self.settings.run_id)

    def _setup_output_dirs(self) -> None:
        base = self._get_output_dir()
        dirs = [
            "step_1_workspace",
            "step_1_workspace/prompt_traces",
            "step_2_time_period",
            "step_2_time_period/week_plans",
            "step_3_decomposition",
            "step_3_decomposition/artifact_specs/meetings",
            "step_3_decomposition/artifact_specs/emails",
            "step_3_decomposition/artifact_specs/documents",
            "step_3_decomposition/artifact_specs/slides",
            "step_4_artifacts/meetings",
            "step_4_artifacts/emails",
            "step_4_artifacts/documents",
            "step_4_artifacts/slides",
            "step_5_events",
            "step_6_postprocess",
            "step_7_speech_prep",
            "step_7_speech_prep/meetings",
            "step_7_speech_prep/speakers",
            "step_8_tts_batch",
            "step_8_tts_batch/meetings",
            "step_9_audio_finalize",
            "step_10_room_simulation",
            "step_11_validation",
            "state",
            "state/snapshots",
        ]
        for d in dirs:
            fileio.makedirs(os.path.join(base, d))

    async def run(self, steps: list[str] | None = None) -> None:
        from .utils.usage_tracker import tracker
        from .utils.version import write_version_file

        start_time = time.time()
        output_dir = self._get_output_dir()
        steps_to_run = steps or ALL_STEPS

        # Forcing a step invalidates everything downstream of it, so force
        # every later step too.
        if self.force_steps:
            first = min(ALL_STEPS.index(s) for s in self.force_steps)
            self.force_steps = set(ALL_STEPS[first:])
            stale = [s for s in ALL_STEPS[first:] if s not in steps_to_run]
            if stale:
                logger.warning(
                    "Forcing %s; downstream steps not selected (%s) keep "
                    "outputs that may now be stale.",
                    ALL_STEPS[first], ", ".join(stale),
                )

        # Clean stale outputs before forced steps (before creating dirs)
        _STEP_DIRS = {
            "step_1": ["step_1_workspace"],
            "step_2": ["step_2_time_period"],
            "step_3": ["step_3_decomposition"],
            "step_4": ["step_4_artifacts", "state"],
            "step_5": ["step_5_events"],
            "step_6": ["step_6_postprocess"],
            "step_7": ["step_7_speech_prep"],
            "step_8": ["step_8_tts_batch"],
            "step_9": ["step_9_audio_finalize"],
            "step_10": ["step_10_room_simulation"],
            "step_11": ["step_11_validation"],
        }
        for step_name in steps_to_run:
            if self._should_force(step_name) and step_name in _STEP_DIRS:
                self._clean_step_output(step_name, _STEP_DIRS[step_name])

        # Create output directories (after cleanup so cleaned dirs get recreated)
        self._setup_output_dirs()

        # Write VERSION file
        config_name = Path(self.config_path).stem
        write_version_file(
            output_dir,
            config_name=config_name,
            run_id=self.settings.run_id,
        )

        # Configure usage tracker (load prior run data for cumulative cost tracking)
        tracker.load_prior(output_dir)
        tracker.set_models(
            llm_model=self.settings.model,
            tts_model=self.settings.tts.model,
        )

        logger.info("Running synthetic data pipeline: %s", ", ".join(steps_to_run))
        logger.info("Output directory: %s", output_dir)

        if "step_1" in steps_to_run:
            tracker.set_step("step_1_workspace")
            await self._run_step_1()

        if "step_2" in steps_to_run:
            tracker.set_step("step_2_time_period")
            await self._run_step_2()

        if "step_3" in steps_to_run:
            tracker.set_step("step_3_decomposition")
            await self._run_step_3()

        if "step_4" in steps_to_run:
            tracker.set_step("step_4_artifacts")
            await self._run_step_4()

        if "step_5" in steps_to_run:
            tracker.set_step("step_5_events")
            await self._run_step_5()

        if "step_6" in steps_to_run:
            tracker.set_step("step_6_postprocess")
            await self._run_step_6()

        if "step_7" in steps_to_run:
            tracker.set_step("step_7_speech_prep")
            await self._run_step_7()

        if "step_8" in steps_to_run:
            tracker.set_step("step_8_tts_batch")
            await self._run_step_8()

        if "step_9" in steps_to_run:
            tracker.set_step("step_9_audio_finalize")
            await self._run_step_9()

        if "step_10" in steps_to_run:
            tracker.set_step("step_10_room_simulation")
            await self._run_step_10()

        if "step_11" in steps_to_run:
            tracker.set_step("step_11_validation")
            self._run_step_11()

        # Save manifest and usage
        self._save_manifest()
        tracker.save(output_dir)
        logger.info("\n%s", tracker.summary())

        elapsed = time.time() - start_time
        logger.info("Pipeline complete in %.1fs", elapsed)

    # ------------------------------------------------------------------
    # Step runners
    # ------------------------------------------------------------------

    async def _run_step_1(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 1: Workspace Sampling")
        logger.info("=" * 60)

        from .steps.step_1_workspace import Step1Workspace

        step = Step1Workspace(self.settings, self._get_output_dir())
        result = await step.run(force=self._should_force("step_1"))
        self._data.update(result)

    async def _run_step_2(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 2: Time Period Planning")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()

        from .steps.step_2_time_period import Step2TimePeriod

        step = Step2TimePeriod(self.settings, self._get_output_dir(), self._data)
        result = await step.run(force=self._should_force("step_2"))
        self._data.update(result)

    async def _run_step_3(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 3: Decomposition")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()
        if "narrative" not in self._data:
            self._load_step_2_data()

        from .steps.step_3_decomposition import Step3Decomposition

        step = Step3Decomposition(self.settings, self._get_output_dir(), self._data)
        result = await step.run(force=self._should_force("step_3"))
        self._data.update(result)

    async def _run_step_4(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 4: Artifact Generation")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()
        if "narrative" not in self._data:
            self._load_step_2_data()
        if "generation_order" not in self._data:
            self._load_step_3_data()

        from .steps.step_4_artifacts import Step4Artifacts

        step = Step4Artifacts(self.settings, self._get_output_dir(), self._data)
        result = await step.run(force=self._should_force("step_4"))
        self._data.update(result)

    async def _run_step_5(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 5: Event Reconciliation")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()
        if "generation_order" not in self._data:
            self._load_step_3_data()
        if "artifacts" not in self._data:
            self._load_step_4_data()

        from .steps.step_5_events import Step5Events

        step = Step5Events(self.settings, self._get_output_dir(), self._data)
        result = await step.run(force=self._should_force("step_5"))
        self._data.update(result)

    async def _run_step_6(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 6: Post-Processing")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()
        if "final_state" not in self._data:
            self._load_step_5_data()
        if "artifacts" not in self._data:
            self._load_step_4_data()

        from .steps.step_6_postprocess import Step6PostProcess

        step = Step6PostProcess(self.settings, self._get_output_dir(), self._data)
        result = await step.run(force=self._should_force("step_6"))
        self._data.update(result)

    async def _run_step_7(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 7: Speech Prep")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()
        if "artifacts" not in self._data:
            self._load_step_4_data()

        from .steps.step_7_speech_prep import Step7SpeechPrep

        step = Step7SpeechPrep(self.settings, self._get_output_dir(), self._data)
        await step.run(force=self._should_force("step_7"))

    async def _run_step_8(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 8: TTS Batch")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()

        from .steps.step_8_tts_batch import Step8TTSBatch

        step = Step8TTSBatch(
            self.settings,
            self._get_output_dir(),
            self._data,
        )
        await step.run(force=self._should_force("step_8"))

    async def _run_step_9(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 9: Audio Finalize")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()
        if "artifacts" not in self._data:
            self._load_step_4_data()

        from .steps.step_9_audio_finalize import Step9AudioFinalize

        step = Step9AudioFinalize(self.settings, self._get_output_dir(), self._data)
        await step.run(force=self._should_force("step_9"))

    async def _run_step_10(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 10: Room Simulation")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()
        if "artifacts" not in self._data:
            self._load_step_4_data()
        # Step 10 reads per-speaker WAV paths and audio metadata that step 9
        # mirrors onto each artifact at the end of its run. When step 10 is
        # invoked standalone (without step 9 in the same process), those
        # fields aren't present in-memory and must be reloaded from disk.
        self._load_step_9_data()

        from .steps.step_10_room_simulation import Step10RoomSimulation

        step = Step10RoomSimulation(self.settings, self._get_output_dir(), self._data)
        await step.run(force=self._should_force("step_10"))

    def _run_step_11(self) -> None:
        logger.info("=" * 60)
        logger.info("STEP 11: Validation")
        logger.info("=" * 60)

        if "workspace_config" not in self._data:
            self._load_step_1_data()
        if "final_state" not in self._data:
            self._load_step_5_data()

        from .steps.step_11_validation import Step11Validation

        step = Step11Validation(self.settings, self._get_output_dir(), self._data)
        step.run()

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_step_1_data(self) -> None:
        base = os.path.join(self._get_output_dir(), "step_1_workspace")

        from .models.calendar import RecurringMeeting
        from .models.people import OrgStructure
        from .models.state import Project, WorkspaceState
        from .models.workspace import WorkspaceConfig

        self._data["workspace_config"] = WorkspaceConfig.model_validate(
            fileio.read_json(os.path.join(base, "workspace_config.json"))
        )
        self._data["org"] = OrgStructure.model_validate(
            fileio.read_json(os.path.join(base, "org.json"))
        )

        projects_data = fileio.read_json(os.path.join(base, "projects.json"))
        if isinstance(projects_data, dict):
            projects_data = list(projects_data.values())
        self._data["projects"] = {
            p["id"]: Project.model_validate(p) for p in projects_data
        }

        self._data["recurring_meetings"] = [
            RecurringMeeting.model_validate(rm)
            for rm in fileio.read_json(os.path.join(base, "recurring_meetings.json"))
        ]

        self._data["initial_state"] = WorkspaceState.model_validate(
            fileio.read_json(os.path.join(base, "workspace_state.json"))
        )
        self._data["voice_assignments"] = fileio.read_json(
            os.path.join(base, "voice_assignments.json")
        )

        logger.info("Loaded step 1 data from disk.")

    def _load_step_2_data(self) -> None:
        base = os.path.join(self._get_output_dir(), "step_2_time_period")
        from .models.planning import Narrative, WeekPlan

        self._data["narrative"] = Narrative.model_validate(
            fileio.read_json(os.path.join(base, "narrative.json"))
        ).model_dump(mode="json")

        week_plans = []
        week_dir = os.path.join(base, "week_plans")
        if os.path.exists(week_dir):
            for fname in sorted(os.listdir(week_dir)):
                if fname.endswith(".json"):
                    week_plans.append(
                        WeekPlan.model_validate(
                            fileio.read_json(os.path.join(week_dir, fname))
                        ).model_dump(mode="json")
                    )
        self._data["week_plans"] = week_plans

        logger.info("Loaded step 2 data from disk.")

    def _load_step_3_data(self) -> None:
        base = os.path.join(self._get_output_dir(), "step_3_decomposition")

        from .models.calendar import Schedule

        self._data["schedule"] = Schedule.model_validate(
            fileio.read_json(os.path.join(base, "calendar.json"))
        )
        self._data["generation_order"] = fileio.read_json(
            os.path.join(base, "generation_order.json")
        )

        graph_path = os.path.join(base, "artifact_graph.json")
        if os.path.exists(graph_path):
            self._data["artifact_graph"] = fileio.read_json(graph_path)

        for spec_type, subdir in [
            ("meeting_specs", "meetings"), ("email_specs", "emails"),
            ("document_specs", "documents"), ("slide_specs", "slides"),
        ]:
            specs = []
            spec_dir = os.path.join(base, "artifact_specs", subdir)
            if os.path.exists(spec_dir):
                for fname in sorted(os.listdir(spec_dir)):
                    if fname.endswith(".json"):
                        specs.append(fileio.read_json(os.path.join(spec_dir, fname)))
            self._data[spec_type] = specs

        logger.info("Loaded step 3 data from disk.")

    def _load_step_4_data(self) -> None:
        base = os.path.join(self._get_output_dir(), "step_4_artifacts")
        from .models.artifacts import Document, EmailThread, MeetingArtifact, SlidePresentation

        artifact_models = {
            "meeting": MeetingArtifact,
            "email_thread": EmailThread,
            "document": Document,
            "slides": SlidePresentation,
        }

        artifacts: dict[str, dict] = {}
        events_by_artifact: dict[str, list] = {}

        # Meetings are stored in subdirectories: meetings/<id>/metadata.json
        meetings_dir = os.path.join(base, "meetings")
        if os.path.exists(meetings_dir):
            for name in sorted(os.listdir(meetings_dir)):
                meta_path = os.path.join(meetings_dir, name, "metadata.json")
                if os.path.isfile(meta_path):
                    data = artifact_models["meeting"].model_validate(
                        fileio.read_json(meta_path)
                    ).model_dump(mode="json")
                    aid = data.get("id", name)
                    artifacts[aid] = data
                    if "events" in data:
                        events_by_artifact[aid] = data["events"]

        # Other types are flat: <type>/<id>.json
        for artifact_type in ("emails", "documents", "slides"):
            type_dir = os.path.join(base, artifact_type)
            if not os.path.exists(type_dir):
                continue
            for fname in sorted(os.listdir(type_dir)):
                if not fname.endswith(".json"):
                    continue
                data = artifact_models[
                    {"emails": "email_thread", "documents": "document", "slides": "slides"}[artifact_type]
                ].model_validate(
                    fileio.read_json(os.path.join(type_dir, fname))
                ).model_dump(mode="json")
                aid = data.get("id", fname.replace(".json", ""))
                artifacts[aid] = data
                if "events" in data:
                    events_by_artifact[aid] = data["events"]

        # Events are stored separately
        events_path = os.path.join(self._get_output_dir(), "state", "generated_events_by_artifact.json")
        if os.path.exists(events_path):
            events_by_artifact = fileio.read_json(events_path)

        self._data["artifacts"] = artifacts
        self._data["generated_events_by_artifact"] = events_by_artifact
        logger.info("Loaded step 4 data: %d artifacts, %d with events", len(artifacts), len(events_by_artifact))

    def _load_step_9_data(self) -> None:
        """Mirror step 9's per-meeting audio paths back onto artifacts.

        Reads ``step_9_audio_finalize/<meeting_id>/`` for every meeting
        recorded in ``audio_results.json`` and populates ``audio_path``,
        ``speaker_audio_paths``, ``timing_data``, and ``rttm_path`` on each
        in-memory artifact. This mirrors what ``Step9AudioFinalize.run()``
        does in-process, but is needed when steps 10 / 11 are invoked
        without re-running step 9 in the same process.
        """
        if "artifacts" not in self._data:
            return
        from .steps.step_9_audio_finalize import Step9AudioFinalize

        step9 = Step9AudioFinalize(self.settings, self._get_output_dir(), self._data)
        loaded = step9._load_from_disk().get("audio_results", {})
        if not loaded:
            logger.info("No step 9 audio_results.json on disk; nothing to load.")
            return
        mirrored = 0
        for meeting_id, info in loaded.items():
            artifact = self._data["artifacts"].get(meeting_id)
            if artifact is None:
                continue
            artifact["audio_path"] = info.get("audio_path")
            artifact["speaker_audio_paths"] = info.get("speaker_audio_paths", {})
            artifact["timing_data"] = info.get("timing_data")
            artifact["rttm_path"] = info.get("rttm_path")
            mirrored += 1
        logger.info("Loaded step 9 audio data for %d meetings from disk.", mirrored)

    def _load_step_5_data(self) -> None:
        base = os.path.join(self._get_output_dir(), "step_5_events")

        from .models.state import EventLog, WorkspaceState

        self._data["final_state"] = WorkspaceState.model_validate(
            fileio.read_json(os.path.join(base, "reconciled_state.json"))
        )
        self._data["event_log"] = EventLog.model_validate(
            fileio.read_json(os.path.join(base, "event_log.json"))
        )

        events_path = os.path.join(base, "events_by_artifact.json")
        if os.path.exists(events_path):
            self._data["generated_events_by_artifact"] = fileio.read_json(events_path)

        logger.info("Loaded step 5 data from disk.")

    def _save_manifest(self) -> None:
        manifest = {
            "output_dir": self._get_output_dir(),
            "run_id": self.settings.run_id,
            "model": self.settings.model,
            "workspace_config": "step_1_workspace/workspace_config.json",
            "org_structure": "step_1_workspace/org.json",
        }

        manifest_path = os.path.join(self._get_output_dir(), "manifest.json")
        fileio.write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(
        description="OtterSynth meeting-corpus generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m osynth.main                          # Run smoke config (text only)
  python -m osynth.main --step step_1 step_2     # Run only steps 1 and 2
  python -m osynth.main --config osynth/config/example_14d.yaml
  python -m osynth.main --force                  # Regenerate everything
  python -m osynth.main --force-step step_6       # Regenerate step 6 and later steps
        """,
    )
    parser.add_argument(
        "--step", nargs="+", choices=ALL_STEPS,
        help="Which steps to run (default: all)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML file (default: bundled text-only smoke config)",
    )
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--max-speakers", type=int, help="Override max speakers")
    parser.add_argument("--run-id", help="Override run ID")
    parser.add_argument("--force", action="store_true", help="Regenerate all outputs")
    parser.add_argument(
        "--force-step", nargs="+", choices=ALL_STEPS,
        help="Regenerate these steps and every later step; use --step to limit which steps run",
    )

    args = parser.parse_args()
    pipeline = SyntheticDataPipeline(args.config)

    if args.seed is not None:
        pipeline.settings.seed = args.seed
    if args.max_speakers is not None:
        pipeline.settings.workspace.max_speakers = args.max_speakers
    if args.run_id:
        pipeline.settings.run_id = args.run_id
    pipeline.force = args.force
    pipeline.force_steps = set(args.force_step or [])

    asyncio.run(pipeline.run(args.step))


if __name__ == "__main__":
    main()
