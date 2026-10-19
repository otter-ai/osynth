"""Step 5: Event reconciliation — replay artifact events to produce final state.

Collects all events emitted by Step 4 artifacts, sorts chronologically,
and replays through the reducer to produce the canonical final workspace
state. Also builds artifact alignment data (e.g., slide timing intervals).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.state import EventLog, EventType, WorkspaceEvent, WorkspaceState, reduce
from ..utils import fileio

logger = logging.getLogger(__name__)


class Step5Events:
    """Replay artifact events to produce final workspace state."""

    def __init__(self, settings: Any, output_root: str, data: dict) -> None:
        self.settings = settings
        self.data = data
        self.output_dir = Path(output_root) / "step_5_events"
        self.initial_state: WorkspaceState = data["initial_state"]
        self.events_by_artifact: dict[str, list[dict]] = data.get("generated_events_by_artifact", {})
        self.specs_by_id: dict[str, dict] = {}
        for key in ("meeting_specs", "email_specs", "document_specs", "slide_specs"):
            for spec in data.get(key, []):
                self.specs_by_id[spec["id"]] = spec

    async def run(self, force: bool = False) -> dict:
        state_path = self.output_dir / "reconciled_state.json"
        if state_path.exists() and not force:
            logger.info("Step 5 outputs exist, loading from disk.")
            return self._load_from_disk()

        # Collect and order all events
        ordered_events = self._ordered_events()
        logger.info("Replaying %d events from %d artifacts",
                     len(ordered_events), len(self.events_by_artifact))

        # Replay through reducer
        event_log = EventLog()
        state = self.initial_state.model_copy(deep=True)

        for event in ordered_events:
            event_log.append(event)
            state = reduce(state, event)

        # Build artifact alignment (slide timing intervals)
        artifact_alignment = self._build_artifact_alignment()

        # Persist
        self._save_all(state, event_log, artifact_alignment)

        return {
            "final_state": state,
            "event_log": event_log,
            "generated_events_by_artifact": self.events_by_artifact,
            "artifact_alignment": artifact_alignment,
        }

    def _ordered_events(self) -> list[WorkspaceEvent]:
        """Flatten and sort all events chronologically."""
        flat: list[tuple[str, int, dict]] = []
        for artifact_id, events in self.events_by_artifact.items():
            for idx, event_dict in enumerate(events):
                flat.append((artifact_id, idx, event_dict))

        # Sort by timestamp, then artifact_id, then index for determinism
        flat.sort(key=lambda x: (x[2].get("timestamp", ""), x[0], x[1]))

        ordered: list[WorkspaceEvent] = []
        for artifact_id, idx, event_dict in flat:
            try:
                event = _coerce_event(event_dict)
                ordered.append(event)
            except Exception as e:
                logger.warning("Skipping malformed event from %s[%d]: %s", artifact_id, idx, e)

        return ordered

    def _build_artifact_alignment(self) -> dict:
        """Build slide timing intervals for meetings with slide dependencies."""
        alignment: dict[str, dict] = {}

        for meeting_id, spec in self.specs_by_id.items():
            if spec.get("type") != "meeting":
                continue
            slide_id = spec.get("related_slide_id")
            if not slide_id:
                continue

            start_str = spec.get("start_time", "")
            duration_min = spec.get("duration_minutes", 30)
            if not start_str:
                continue

            try:
                datetime.fromisoformat(start_str)
            except ValueError:
                continue

            slide_spec = self.specs_by_id.get(slide_id, {})
            num_slides = slide_spec.get("num_slides", 8)

            # Allocate ~60% of meeting time to slides (rest is discussion)
            slide_time_ms = int(duration_min * 60 * 1000 * 0.6)
            per_slide_ms = slide_time_ms // max(num_slides, 1)

            intervals = []
            for i in range(num_slides):
                intervals.append({
                    "slide_number": i + 1,
                    "start_ms": i * per_slide_ms,
                    "end_ms": (i + 1) * per_slide_ms,
                })

            alignment[meeting_id] = {
                "slide_artifact_id": slide_id,
                "num_slides": num_slides,
                "intervals": intervals,
            }

        logger.info("Built alignment for %d meetings with slides", len(alignment))
        return alignment

    def _save_all(self, state: WorkspaceState, event_log: EventLog, alignment: dict) -> None:
        fileio.makedirs(self.output_dir)
        fileio.write_json(self.output_dir / "reconciled_state.json", state.model_dump(mode="json"))
        fileio.write_json(self.output_dir / "event_log.json", event_log.model_dump(mode="json"))
        fileio.write_json(self.output_dir / "events_by_artifact.json", self.events_by_artifact)
        fileio.write_json(self.output_dir / "artifact_alignment.json", alignment)
        logger.info("Step 5 outputs saved to %s", self.output_dir)

    def _load_from_disk(self) -> dict:
        with open(self.output_dir / "reconciled_state.json") as f:
            final_state = WorkspaceState.model_validate(json.load(f))
        with open(self.output_dir / "event_log.json") as f:
            event_log = EventLog.model_validate(json.load(f))

        events_path = self.output_dir / "events_by_artifact.json"
        events_by_artifact = {}
        if events_path.exists():
            with open(events_path) as f:
                events_by_artifact = json.load(f)

        alignment_path = self.output_dir / "artifact_alignment.json"
        alignment = {}
        if alignment_path.exists():
            with open(alignment_path) as f:
                alignment = json.load(f)

        logger.info("Loaded step 5: %d events, final state at %s",
                     len(event_log.events), final_state.timestamp)
        return {
            "final_state": final_state,
            "event_log": event_log,
            "generated_events_by_artifact": events_by_artifact,
            "artifact_alignment": alignment,
        }


def _coerce_event(data: dict) -> WorkspaceEvent:
    """Convert a raw dict to a WorkspaceEvent, handling string enum values."""
    event_type = data.get("type", "")
    if isinstance(event_type, str):
        event_type = EventType(event_type)
    timestamp = data.get("timestamp", "")
    if isinstance(timestamp, str) and timestamp:
        timestamp = datetime.fromisoformat(timestamp)
    elif not timestamp:
        timestamp = datetime.min

    return WorkspaceEvent(
        type=event_type,
        timestamp=timestamp,
        data=data.get("data", {}),
        source_artifact_id=data.get("source_artifact_id"),
    )
