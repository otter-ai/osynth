"""Step 11: Validation — check consistency of all generated workspace data.

Runs a set of validators against the final workspace state, calendar,
org structure, and artifacts. Produces a markdown report and optionally
fails the pipeline if errors are found.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..models.calendar import (
    MeetingJoinMode,
    MeetingJoinSpec,
    ParticipantJoinMode,
    Schedule,
)
from ..models.people import OrgStructure
from ..models.settings import SyntheticDataSettings
from ..models.state import EventLog, WorkspaceState
from ..models.workspace import WorkspaceConfig
from ..tts.truncation_sanitizer import (
    count_single_letter_tts,
    count_word_truncations,
)
from ..utils import fileio

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    validator: str
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "validator": self.validator,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
        }


class Step11Validation:
    """Run validation checks on all generated data."""

    def __init__(self, settings: SyntheticDataSettings, output_root: str, data: dict) -> None:
        self.settings = settings
        self.data = data
        self.output_dir = Path(output_root) / "step_11_validation"
        self.fail_on_error: bool = settings.validation.fail_on_error

        self.workspace_config: WorkspaceConfig = data["workspace_config"]
        self.org: OrgStructure = data["org"]
        self.schedule: Schedule = data.get("schedule", Schedule())
        self.final_state: WorkspaceState = data.get("final_state", WorkspaceState(timestamp=datetime.min))
        self.event_log: EventLog = data.get("event_log", EventLog())
        self.artifacts: dict[str, dict] = data.get("artifacts", {})
        self.room_simulation_enabled: bool = settings.room_simulation.enabled
        self.room_output_dir: Path = Path(output_root) / "step_10_room_simulation"
        self.audio_output_dir: Path = Path(output_root) / "step_9_audio_finalize"
        self.speech_prep_dir: Path = Path(output_root) / "step_7_speech_prep"
        self.artifacts_output_dir: Path = Path(output_root) / "step_4_artifacts"
        self.generate_audio: bool = settings.generation.generate_audio
        self.max_truncation_rate: float = float(settings.tts.max_truncation_rate)
        self.duration_targeting_settings = settings.generation.duration_targeting

    def run(self) -> list[ValidationResult]:
        fileio.makedirs(self.output_dir)

        results: list[ValidationResult] = []
        results.extend(self._validate_scheduling())
        results.extend(self._validate_working_hours())
        results.extend(self._validate_referential_integrity())
        results.extend(self._validate_org_structure())
        results.extend(self._validate_action_item_lifecycle())
        results.extend(self._validate_meeting_join_specs())
        results.extend(self._validate_room_simulation())
        results.extend(self._validate_speech_overlap())
        results.extend(self._validate_short_turn_trim())
        results.extend(self._validate_word_truncations())
        results.extend(self._validate_meeting_duration_fill())

        # Report
        self._save_report(results)

        errors = [r for r in results if r.severity == Severity.ERROR]
        warnings = [r for r in results if r.severity == Severity.WARNING]
        logger.info("Validation complete: %d errors, %d warnings, %d info",
                     len(errors), len(warnings),
                     len(results) - len(errors) - len(warnings))

        if errors and self.fail_on_error:
            report_path = self.output_dir / "validation_report.md"
            raise RuntimeError(
                f"Validation failed with {len(errors)} errors. See {report_path}"
            )

        return results

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    def _validate_scheduling(self) -> list[ValidationResult]:
        """Check for overlapping meetings per person."""
        results: list[ValidationResult] = []

        # Group events by person
        events_by_person: dict[str, list] = {}
        for event in self.schedule.events:
            all_ids = set(event.attendee_ids) | {event.organizer_id}
            for pid in all_ids:
                events_by_person.setdefault(pid, []).append(event)

        for pid, events in events_by_person.items():
            sorted_events = sorted(events, key=lambda e: e.start_time)
            for i in range(len(sorted_events) - 1):
                curr = sorted_events[i]
                nxt = sorted_events[i + 1]
                if curr.end_time > nxt.start_time:
                    overlap_min = (curr.end_time - nxt.start_time).total_seconds() / 60
                    results.append(ValidationResult(
                        validator="scheduling",
                        severity=Severity.ERROR,
                        message=f"Person {pid}: '{curr.title}' overlaps with '{nxt.title}' by {overlap_min:.0f}min",
                        details={"person_id": pid, "event_1": curr.id, "event_2": nxt.id,
                                 "overlap_minutes": round(overlap_min)},
                    ))

        return results

    def _validate_working_hours(self) -> list[ValidationResult]:
        """Check events are within configured working hours."""
        results: list[ValidationResult] = []
        wh_start, wh_end = self.workspace_config.working_hours
        working_days = set(self.workspace_config.working_days)

        for event in self.schedule.events:
            if event.start_time.hour < wh_start or event.end_time.hour > wh_end:
                results.append(ValidationResult(
                    validator="working_hours",
                    severity=Severity.WARNING,
                    message=f"'{event.title}' outside working hours ({event.start_time} - {event.end_time})",
                    details={"event_id": event.id},
                ))
            if event.start_time.weekday() not in working_days:
                results.append(ValidationResult(
                    validator="working_hours",
                    severity=Severity.WARNING,
                    message=f"'{event.title}' on non-working day ({event.start_time.strftime('%A')})",
                    details={"event_id": event.id},
                ))

        return results

    def _validate_referential_integrity(self) -> list[ValidationResult]:
        """Check all ID references are valid."""
        results: list[ValidationResult] = []

        for event in self.schedule.events:
            if not self.org.has_person(event.organizer_id):
                results.append(ValidationResult(
                    validator="referential_integrity",
                    severity=Severity.ERROR,
                    message=f"Event '{event.title}' has unknown organizer {event.organizer_id}",
                    details={"event_id": event.id},
                ))
            for aid in event.attendee_ids:
                if not self.org.has_person(aid):
                    results.append(ValidationResult(
                        validator="referential_integrity",
                        severity=Severity.ERROR,
                        message=f"Event '{event.title}' has unknown attendee {aid}",
                        details={"event_id": event.id, "person_id": aid},
                    ))

        # Check team references
        for tid, team in self.org.teams.items():
            if team.manager_id not in self.org.people:
                results.append(ValidationResult(
                    validator="referential_integrity",
                    severity=Severity.ERROR,
                    message=f"Team '{team.name}' has unknown manager {team.manager_id}",
                    details={"team_id": tid},
                ))

        # Check reporting chain
        for pid, mid in self.org.reporting.items():
            if not self.org.has_person(pid):
                results.append(ValidationResult(
                    validator="referential_integrity",
                    severity=Severity.ERROR,
                    message=f"Reporting: unknown person {pid}",
                ))
            if not self.org.has_person(mid):
                results.append(ValidationResult(
                    validator="referential_integrity",
                    severity=Severity.ERROR,
                    message=f"Reporting: unknown manager {mid} for {pid}",
                ))

        return results

    def _validate_org_structure(self) -> list[ValidationResult]:
        """Check organizational structure consistency."""
        results: list[ValidationResult] = []

        # Check person team assignments
        people_in_teams: set[str] = set()
        for team in self.org.teams.values():
            people_in_teams.update(team.member_ids)

        for pid, person in self.org.people.items():
            if pid not in people_in_teams:
                results.append(ValidationResult(
                    validator="org_structure",
                    severity=Severity.WARNING,
                    message=f"Person {person.name} ({pid}) not in any team's member list",
                ))
            if person.team_id not in self.org.teams:
                results.append(ValidationResult(
                    validator="org_structure",
                    severity=Severity.ERROR,
                    message=f"Person {person.name} references unknown team {person.team_id}",
                    details={"person_id": pid},
                ))

        # Check for reporting cycles (DFS)
        visited: set[str] = set()
        in_stack: set[str] = set()

        def has_cycle(pid: str) -> bool:
            if pid in in_stack:
                return True
            if pid in visited:
                return False
            visited.add(pid)
            in_stack.add(pid)
            manager = self.org.reporting.get(pid)
            if manager and has_cycle(manager):
                return True
            in_stack.discard(pid)
            return False

        for pid in self.org.reporting:
            if has_cycle(pid):
                results.append(ValidationResult(
                    validator="org_structure",
                    severity=Severity.ERROR,
                    message=f"Reporting cycle detected involving {pid}",
                ))
                break  # One cycle report is enough

        # Check voice assignments
        for pid, person in self.org.people.items():
            if not person.voice_id:
                results.append(ValidationResult(
                    validator="org_structure",
                    severity=Severity.WARNING,
                    message=f"Person {person.name} ({pid}) has no voice_id assigned",
                ))

        return results

    def _validate_action_item_lifecycle(self) -> list[ValidationResult]:
        """Check action items are created before completed."""
        results: list[ValidationResult] = []

        created_at: dict[str, datetime] = {}
        completed_at: dict[str, datetime] = {}

        for event in self.event_log.events:
            ai_id = event.data.get("id", "")
            if not ai_id:
                continue
            if event.type.value == "action_item_created":
                created_at[ai_id] = event.timestamp
            elif event.type.value == "action_item_completed":
                completed_at[ai_id] = event.timestamp

        for ai_id, comp_time in completed_at.items():
            if ai_id not in created_at:
                results.append(ValidationResult(
                    validator="action_item_lifecycle",
                    severity=Severity.WARNING,
                    message=f"Action item {ai_id} completed but never created",
                ))
            elif comp_time < created_at[ai_id]:
                results.append(ValidationResult(
                    validator="action_item_lifecycle",
                    severity=Severity.ERROR,
                    message=f"Action item {ai_id} completed before created",
                    details={"created": str(created_at[ai_id]), "completed": str(comp_time)},
                ))

        return results

    def _validate_meeting_join_specs(self) -> list[ValidationResult]:
        """Check per-meeting modality + participant join consistency.

        The join_spec is sampled deterministically in step 3 and threaded
        through step 4. The invariants below should hold for any completed
        meeting; skipped/canceled meetings are skipped here.
        """
        results: list[ValidationResult] = []

        for art_id, art in self.artifacts.items():
            if art.get("type") != "meeting":
                continue
            if art.get("realized_status") != "completed":
                continue

            raw_spec = art.get("join_spec")
            if not raw_spec:
                results.append(ValidationResult(
                    validator="meeting_join_spec",
                    severity=Severity.WARNING,
                    message=f"Meeting '{art.get('title', art_id)}' has no join_spec",
                    details={"meeting_id": art_id},
                ))
                continue

            try:
                spec = MeetingJoinSpec.model_validate(raw_spec)
            except Exception as exc:
                results.append(ValidationResult(
                    validator="meeting_join_spec",
                    severity=Severity.ERROR,
                    message=f"Meeting '{art.get('title', art_id)}' has invalid join_spec: {exc}",
                    details={"meeting_id": art_id},
                ))
                continue

            attendee_ids: list[str] = list(art.get("attendee_ids", []))
            attendee_set = set(attendee_ids)
            planned_changes: dict[str, str] = art.get("planned_attendance_changes", {}) or {}
            missed = {pid for pid, status in planned_changes.items() if status == "missed"}
            active = [pid for pid in attendee_ids if pid not in missed]

            unknown = [pid for pid in spec.participant_modes if pid not in attendee_set]
            if unknown:
                results.append(ValidationResult(
                    validator="meeting_join_spec",
                    severity=Severity.ERROR,
                    message=(
                        f"Meeting '{art.get('title', art_id)}' join_spec references "
                        f"unknown attendees: {sorted(unknown)}"
                    ),
                    details={"meeting_id": art_id, "unknown_ids": sorted(unknown)},
                ))

            missing = [pid for pid in active if pid not in spec.participant_modes]
            if missing:
                results.append(ValidationResult(
                    validator="meeting_join_spec",
                    severity=Severity.WARNING,
                    message=(
                        f"Meeting '{art.get('title', art_id)}' is missing join info for "
                        f"non-missed attendees: {sorted(missing)}"
                    ),
                    details={"meeting_id": art_id, "missing_ids": sorted(missing)},
                ))

            participant_modes_for_active = {
                pid: info for pid, info in spec.participant_modes.items() if pid in active
            }
            modes_present = {info.mode for info in participant_modes_for_active.values()}

            if spec.mode == MeetingJoinMode.VIRTUAL:
                in_room = [pid for pid, info in participant_modes_for_active.items()
                           if info.mode == ParticipantJoinMode.IN_ROOM]
                if in_room:
                    results.append(ValidationResult(
                        validator="meeting_join_spec",
                        severity=Severity.ERROR,
                        message=(
                            f"Virtual meeting '{art.get('title', art_id)}' has in_room "
                            f"participants: {sorted(in_room)}"
                        ),
                        details={"meeting_id": art_id, "in_room_ids": sorted(in_room)},
                    ))
            elif spec.mode == MeetingJoinMode.IN_PERSON:
                remote = [pid for pid, info in participant_modes_for_active.items()
                          if info.mode == ParticipantJoinMode.REMOTE]
                if remote:
                    results.append(ValidationResult(
                        validator="meeting_join_spec",
                        severity=Severity.ERROR,
                        message=(
                            f"In-person meeting '{art.get('title', art_id)}' has remote "
                            f"participants: {sorted(remote)}"
                        ),
                        details={"meeting_id": art_id, "remote_ids": sorted(remote)},
                    ))
            elif spec.mode == MeetingJoinMode.HYBRID:
                if len(active) >= 2:
                    has_in_room = ParticipantJoinMode.IN_ROOM in modes_present
                    has_remote = ParticipantJoinMode.REMOTE in modes_present
                    if not (has_in_room and has_remote):
                        results.append(ValidationResult(
                            validator="meeting_join_spec",
                            severity=Severity.ERROR,
                            message=(
                                f"Hybrid meeting '{art.get('title', art_id)}' with "
                                f"{len(active)} active attendees lacks both in_room "
                                f"and remote participants (modes present: "
                                f"{sorted(m.value for m in modes_present)})"
                            ),
                            details={"meeting_id": art_id},
                        ))

        return results

    def _validate_room_simulation(self) -> list[ValidationResult]:
        """When room simulation is enabled, check expected outputs exist."""
        results: list[ValidationResult] = []
        if not self.room_simulation_enabled:
            return results

        for art_id, art in self.artifacts.items():
            if art.get("type") != "meeting":
                continue
            if art.get("realized_status") != "completed":
                continue
            raw_spec = art.get("join_spec")
            if not raw_spec:
                continue
            try:
                spec = MeetingJoinSpec.model_validate(raw_spec)
            except Exception:
                continue

            has_in_room = any(
                info.mode == ParticipantJoinMode.IN_ROOM
                for info in spec.participant_modes.values()
            )
            if not has_in_room:
                continue

            combined = self.room_output_dir / art_id / "combined_room.wav"
            if not combined.exists():
                results.append(ValidationResult(
                    validator="room_simulation",
                    severity=Severity.ERROR,
                    message=(
                        f"Meeting '{art.get('title', art_id)}' has in-room "
                        f"participants but {combined.name} is missing"
                    ),
                    details={"meeting_id": art_id, "expected_path": str(combined)},
                ))

            combined_full = self.room_output_dir / art_id / "combined_room_full.wav"
            if not combined_full.exists():
                results.append(ValidationResult(
                    validator="room_simulation",
                    severity=Severity.ERROR,
                    message=(
                        f"Meeting '{art.get('title', art_id)}' has in-room "
                        f"participants but {combined_full.name} is missing"
                    ),
                    details={
                        "meeting_id": art_id,
                        "expected_path": str(combined_full),
                    },
                ))

        return results

    def _validate_speech_overlap(self) -> list[ValidationResult]:
        """Assert no per-speaker self-overlap; log overall overlap ratio.

        Reads each completed meeting's timing.json from step 9. Skips silently
        when audio generation is disabled or timing files are missing.
        """
        results: list[ValidationResult] = []
        if not self.generate_audio:
            return results

        epsilon_ms = 50.0  # match mixer's monotonic sweep tolerance
        total_speech_ms = 0.0
        total_pairwise_overlap_ms = 0.0
        total_applied = 0
        total_requested = 0
        total_dropped_placement = 0
        total_missing_word_alignment = 0
        total_dropped_self = 0
        total_bc_turns = 0
        total_bc_requested = 0
        total_bc_applied = 0
        total_bc_unplaced = 0
        meetings_checked = 0

        for art_id, art in self.artifacts.items():
            if art.get("type") != "meeting":
                continue
            if art.get("realized_status") != "completed":
                continue

            timing_path = self.audio_output_dir / art_id / "timing.json"
            if not timing_path.exists():
                continue

            try:
                timing = fileio.read_json(timing_path)
            except Exception as exc:
                results.append(ValidationResult(
                    validator="speech_overlap",
                    severity=Severity.WARNING,
                    message=f"Could not read timing.json for '{art.get('title', art_id)}': {exc}",
                    details={"meeting_id": art_id, "path": str(timing_path)},
                ))
                continue

            meetings_checked += 1
            turns = timing.get("turns", []) or []
            stats = timing.get("overlap_stats") or {}
            total_requested += int(stats.get("requested", 0))
            total_applied += int(stats.get("applied", 0))
            total_dropped_placement += int(stats.get("dropped_at_placement", 0))
            total_missing_word_alignment += int(
                stats.get("overlap_anchor_missing_word_alignment", 0)
            )
            total_dropped_self += int(stats.get("dropped_self_overlap", 0))
            total_bc_requested += int(stats.get("requested_backchannel", 0))
            total_bc_applied += int(stats.get("applied_backchannel", 0))

            for turn in turns:
                if not turn.get("is_backchannel"):
                    continue
                total_bc_turns += 1
                has_anchor_info = (
                    turn.get("overlap_anchor_requested") is not None
                    or turn.get("overlap_anchor_resolved") is not None
                )
                if not has_anchor_info:
                    total_bc_unplaced += 1

            by_speaker: dict[str, list[tuple[float, float]]] = {}
            timeline: list[tuple[float, float]] = []
            for turn in turns:
                start = turn.get("speech_start_ms")
                end = turn.get("speech_end_ms")
                if start is None or end is None:
                    start = turn.get("start_time_ms")
                    end = turn.get("end_time_ms")
                spk = turn.get("speaker_key") or turn.get("speaker") or turn.get("speaker_id")
                if start is None or end is None or spk is None:
                    continue
                if float(end) <= float(start):
                    continue
                by_speaker.setdefault(str(spk), []).append((float(start), float(end)))
                timeline.append((float(start), float(end)))

            for spk, segs in by_speaker.items():
                segs.sort(key=lambda s: s[0])
                for i in range(len(segs) - 1):
                    a_start, a_end = segs[i]
                    b_start, b_end = segs[i + 1]
                    if b_start + epsilon_ms < a_end:
                        overlap_ms = a_end - b_start
                        results.append(ValidationResult(
                            validator="speech_overlap",
                            severity=Severity.ERROR,
                            message=(
                                f"Meeting '{art.get('title', art_id)}': speaker {spk} "
                                f"self-overlaps by {overlap_ms:.0f}ms"
                            ),
                            details={
                                "meeting_id": art_id,
                                "speaker": spk,
                                "overlap_ms": round(overlap_ms, 1),
                                "seg_a": [a_start, a_end],
                                "seg_b": [b_start, b_end],
                            },
                        ))

            # Pairwise cross-speaker overlap for ratio reporting
            timeline.sort(key=lambda s: s[0])
            for (a_start, a_end), (b_start, b_end) in zip(timeline, timeline[1:]):
                if b_start < a_end:
                    total_pairwise_overlap_ms += min(a_end, b_end) - b_start
                total_speech_ms += a_end - a_start
            if timeline:
                last_start, last_end = timeline[-1]
                total_speech_ms += last_end - last_start

        if meetings_checked > 0:
            ratio = (
                total_pairwise_overlap_ms / total_speech_ms
                if total_speech_ms > 0 else 0.0
            )
            bc_placement_rate = (
                total_bc_applied / total_bc_requested
                if total_bc_requested > 0 else 0.0
            )
            results.append(ValidationResult(
                validator="speech_overlap",
                severity=Severity.INFO,
                message=(
                    f"Overlap stats across {meetings_checked} meetings: "
                    f"applied {total_applied}/{total_requested} requested, "
                    f"dropped {total_dropped_placement} at placement and "
                    f"{total_missing_word_alignment} for missing anchor words, "
                    f"{total_dropped_self} for self-overlap. "
                    f"Cross-speaker overlap ratio: {ratio:.3f}. "
                    f"Backchannels: {total_bc_turns} total, "
                    f"{total_bc_applied}/{total_bc_requested} placed at LLM anchor "
                    f"(rate {bc_placement_rate:.2f})"
                ),
                details={
                    "meetings_checked": meetings_checked,
                    "requested": total_requested,
                    "applied": total_applied,
                    "dropped_at_placement": total_dropped_placement,
                    "overlap_anchor_missing_word_alignment": total_missing_word_alignment,
                    "dropped_self_overlap": total_dropped_self,
                    "overlap_ratio": round(ratio, 4),
                    "backchannel_turns": total_bc_turns,
                    "backchannel_requested": total_bc_requested,
                    "backchannel_applied": total_bc_applied,
                    "backchannel_unplaced": total_bc_unplaced,
                },
            ))
            if total_bc_unplaced:
                results.append(ValidationResult(
                    validator="speech_overlap",
                    severity=Severity.WARNING,
                    message=(
                        f"{total_bc_unplaced} backchannel turn(s) have no "
                        f"overlap_anchor_requested or overlap_anchor_resolved "
                        f"— BC anchor sub-stage may have failed silently"
                    ),
                    details={"backchannel_unplaced": total_bc_unplaced},
                ))
            if total_missing_word_alignment:
                results.append(ValidationResult(
                    validator="speech_overlap",
                    severity=Severity.WARNING,
                    message=(
                        f"{total_missing_word_alignment} requested overlap anchor(s) "
                        f"lacked word alignments; hybrid alignment may have "
                        f"under-selected NFA turns"
                    ),
                    details={
                        "overlap_anchor_missing_word_alignment": total_missing_word_alignment,
                    },
                ))

        return results

    def _validate_short_turn_trim(self) -> list[ValidationResult]:
        """Report short-turn (padded) trim counters from step 9.

        Padded turns synthesize repeated copies of the original text; the
        alignment-driven trim then keeps only the first-repetition word
        alignments so the mixer clips back to one audible occurrence.
        Cases where NFA failed to return enough words to do the trim fall
        through to the un-trimmed path (full audio plays). A high fallback
        rate signals NFA quality problems on short audio.
        """
        results: list[ValidationResult] = []
        if not self.generate_audio:
            return results

        totals = {
            "padded": 0,
            "trimmed": 0,
            "trim_word_count_mismatch": 0,
            "energy_fallback_padded": 0,
            "hybrid_nfa_entries": 0,
            "hybrid_nfa_short_padding_trim": 0,
            "hybrid_nfa_future_overlap_anchor": 0,
            "hybrid_nfa_both_reasons": 0,
            "hybrid_energy_only_turns": 0,
        }
        meetings_checked = 0
        for art_id, art in self.artifacts.items():
            if art.get("type") != "meeting":
                continue
            if art.get("realized_status") != "completed":
                continue
            timing_path = self.audio_output_dir / art_id / "timing.json"
            if not timing_path.exists():
                continue
            try:
                timing = fileio.read_json(timing_path)
            except Exception:
                continue
            meetings_checked += 1
            stats = timing.get("short_turn_stats") or {}
            for key in totals:
                totals[key] += int(stats.get(key, 0))

        if meetings_checked == 0 or totals["padded"] == 0:
            return results

        fallback = (
            totals["trim_word_count_mismatch"] + totals["energy_fallback_padded"]
        )
        fallback_rate = fallback / totals["padded"] if totals["padded"] else 0.0
        results.append(ValidationResult(
            validator="short_turn_trim",
            severity=Severity.INFO,
            message=(
                f"Short-turn trim across {meetings_checked} meetings: "
                f"{totals['padded']} padded turn(s), "
                f"{totals['trimmed']} trimmed via NFA, "
                f"{totals['trim_word_count_mismatch']} NFA word-count mismatch, "
                f"{totals['energy_fallback_padded']} energy-fallback, "
                f"{totals['hybrid_nfa_future_overlap_anchor']} hybrid future-anchor "
                f"NFA turn(s). "
                f"Fallback rate {fallback_rate:.2%}."
            ),
            details={
                "meetings_checked": meetings_checked,
                **totals,
                "fallback_rate": round(fallback_rate, 4),
            },
        ))
        if fallback_rate > 0.05:
            results.append(ValidationResult(
                validator="short_turn_trim",
                severity=Severity.WARNING,
                message=(
                    f"Short-turn trim fallback rate {fallback_rate:.2%} > 5% — "
                    f"NFA likely struggling on short audio; review padded turns"
                ),
                details={"fallback_rate": round(fallback_rate, 4), **totals},
            ))
        return results

    def _validate_word_truncations(self) -> list[ValidationResult]:
        """Count single-letter and total word-truncation fragments in ``tts_text``.

        The single-letter counter must be zero on every freshly generated
        run — the prompt forbids single-letter fragments and the post-LLM
        sanitizer strips any that slip through. A non-zero count is a hard
        error signaling that both layers failed simultaneously.

        The aggregate truncation rate per 100 spoken-form words gives the
        prompt + sanitizer feedback loop a single observability number. A
        rate above 150% of the configured cap raises a warning (50% over
        budget); much higher means the LLM is over-emitting truncations
        and the sanitizer's rate cap is doing structural work.
        """
        results: list[ValidationResult] = []
        meetings_dir = self.speech_prep_dir / "meetings"
        if not meetings_dir.exists():
            return results

        single_letter_total = 0
        truncation_total = 0
        total_words = 0
        meetings_checked = 0
        per_meeting_hits: list[tuple[str, int, int]] = []

        for meeting_dir in sorted(meetings_dir.iterdir()):
            if not meeting_dir.is_dir():
                continue
            ft_path = meeting_dir / "formatted_turns.json"
            if not ft_path.exists():
                continue
            try:
                turns = fileio.read_json(ft_path)
            except Exception:
                continue
            meetings_checked += 1
            meeting_single_letter = 0
            meeting_truncation = 0
            for turn in turns or []:
                tts_text = str(turn.get("tts_text") or turn.get("text") or "")
                meeting_single_letter += count_single_letter_tts(tts_text)
                meeting_truncation += count_word_truncations(tts_text)
                total_words += len(tts_text.split())
            if meeting_single_letter or meeting_truncation:
                per_meeting_hits.append(
                    (meeting_dir.name, meeting_single_letter, meeting_truncation),
                )
            single_letter_total += meeting_single_letter
            truncation_total += meeting_truncation

        if not meetings_checked:
            return results

        rate_per_100 = (
            (truncation_total / total_words) * 100.0 if total_words > 0 else 0.0
        )
        warn_threshold = self.max_truncation_rate * 1.5 if self.max_truncation_rate > 0 else 1.5
        per_meeting_offenders = [
            (mid, sl, tr) for mid, sl, tr in per_meeting_hits if sl > 0
        ]

        severity = (
            Severity.ERROR if single_letter_total > 0 else Severity.INFO
        )
        results.append(ValidationResult(
            validator="word_truncations",
            severity=severity,
            message=(
                f"Word-truncation counters across {meetings_checked} meetings: "
                f"single-letter fragments {single_letter_total} "
                f"(target=0), total truncations {truncation_total}, "
                f"rate {rate_per_100:.2f}/100 words "
                f"(cap {self.max_truncation_rate:.2f}/100, "
                f"warn>{warn_threshold:.2f})"
            ),
            details={
                "meetings_checked": meetings_checked,
                "single_letter_truncation_count": single_letter_total,
                "truncation_total": truncation_total,
                "total_words": total_words,
                "truncation_rate_per_100_words": round(rate_per_100, 4),
                "max_truncation_rate": self.max_truncation_rate,
                "per_meeting_single_letter_offenders": [
                    {"meeting_id": mid, "single_letter": sl, "total": tr}
                    for mid, sl, tr in per_meeting_offenders[:10]
                ],
            },
        ))

        if rate_per_100 > warn_threshold:
            results.append(ValidationResult(
                validator="word_truncations",
                severity=Severity.WARNING,
                message=(
                    f"Truncation rate {rate_per_100:.2f}/100 words exceeds "
                    f"{warn_threshold:.2f}/100 (1.5× cap of "
                    f"{self.max_truncation_rate:.2f}/100). The prompt or "
                    f"sanitizer rate cap may be drifting; review recent "
                    f"runs."
                ),
                details={
                    "rate_per_100_words": round(rate_per_100, 4),
                    "warn_threshold": round(warn_threshold, 4),
                    "max_truncation_rate": self.max_truncation_rate,
                },
            ))

        return results

    def _validate_meeting_duration_fill(self) -> list[ValidationResult]:
        """Compare audio (or transcript) length against the scheduled slot.

        Reads ``timing.json::total_duration_ms`` from step 9 when audio is
        generated, else falls back to a word-count proxy from the meeting
        artifact's transcript. The expected length comes from the calendar
        slot (``end_time - start_time`` on the meeting artifact). Writes an
        aggregate ``duration_fill.json`` next to the markdown report with
        mean / median / p10 fill ratios and the list of offending meetings.

        The shortfall threshold is anchored to
        ``duration_targeting.min_words_fraction``: anything below
        ``0.75 * min_words_fraction`` of the scheduled slot is a warning
        (default ~60% slot fill). When duration targeting is disabled the
        validator still produces a report but does not raise warnings.
        """
        results: list[ValidationResult] = []

        cfg = self.duration_targeting_settings

        wpm_min, _wpm_max = 120, 150  # English fallback (cf. language.py).
        from ..utils.language import LANG_CONFIG  # local import to avoid cycle

        lang_cfg = LANG_CONFIG.get(self.settings.language, LANG_CONFIG["en"])
        try:
            lo_str, hi_str = str(
                lang_cfg.get("words_per_minute_range", "120-150")
            ).split("-", 1)
            wpm_min = int(lo_str.strip())
        except (ValueError, AttributeError):
            pass

        fill_ratios: list[float] = []
        per_meeting: list[dict] = []
        warn_below = max(0.0, 0.75 * cfg.min_words_fraction) if cfg.enabled else 0.0
        floor_target = cfg.min_words_fraction if cfg.enabled else 0.0

        for art_id, art in self.artifacts.items():
            if art.get("type") != "meeting":
                continue
            if art.get("realized_status") != "completed":
                continue
            start = art.get("start_time")
            end = art.get("end_time")
            if not start or not end:
                continue
            try:
                start_dt = (
                    start if isinstance(start, datetime) else datetime.fromisoformat(str(start))
                )
                end_dt = (
                    end if isinstance(end, datetime) else datetime.fromisoformat(str(end))
                )
            except (TypeError, ValueError):
                continue
            scheduled_seconds = (end_dt - start_dt).total_seconds()
            if scheduled_seconds <= 0:
                continue
            scheduled_minutes = scheduled_seconds / 60.0

            actual_seconds: float | None = None
            source = "audio"
            if self.generate_audio:
                timing_path = self.audio_output_dir / art_id / "timing.json"
                if timing_path.exists():
                    try:
                        timing = fileio.read_json(timing_path)
                        total_ms = timing.get("total_duration_ms")
                        if total_ms is not None:
                            actual_seconds = float(total_ms) / 1000.0
                    except Exception:
                        pass

            if actual_seconds is None:
                # Word-count proxy when audio is missing / disabled: assume
                # the language's lower WPM bound (conservative — slightly
                # over-estimates duration so we don't falsely flag).
                transcript = art.get("transcript_json") or []
                word_count = sum(
                    len(str(t.get("text") or "").split()) for t in transcript
                )
                if word_count <= 0:
                    continue
                actual_seconds = (word_count / max(1, wpm_min)) * 60.0
                source = "transcript_wpm_proxy"

            fill_ratio = (actual_seconds / 60.0) / scheduled_minutes
            fill_ratios.append(fill_ratio)
            per_meeting.append({
                "meeting_id": art_id,
                "title": art.get("title", ""),
                "scheduled_minutes": round(scheduled_minutes, 2),
                "actual_minutes": round(actual_seconds / 60.0, 2),
                "fill_ratio": round(fill_ratio, 4),
                "source": source,
            })

            if cfg.enabled and fill_ratio < warn_below:
                results.append(ValidationResult(
                    validator="meeting_duration_fill",
                    severity=Severity.WARNING,
                    message=(
                        f"Meeting '{art.get('title', art_id)}' filled only "
                        f"{fill_ratio:.0%} of its {scheduled_minutes:.0f}-min "
                        f"calendar slot ({actual_seconds / 60.0:.1f} min "
                        f"actual via {source})"
                    ),
                    details={
                        "meeting_id": art_id,
                        "scheduled_minutes": round(scheduled_minutes, 2),
                        "actual_minutes": round(actual_seconds / 60.0, 2),
                        "fill_ratio": round(fill_ratio, 4),
                        "source": source,
                        "warn_below": round(warn_below, 4),
                    },
                ))

        if not fill_ratios:
            return results

        sorted_ratios = sorted(fill_ratios)
        n = len(sorted_ratios)
        mean = sum(sorted_ratios) / n
        median = sorted_ratios[n // 2] if n % 2 == 1 else (
            (sorted_ratios[n // 2 - 1] + sorted_ratios[n // 2]) / 2
        )
        p10 = sorted_ratios[max(0, int(n * 0.10) - 1)] if n >= 10 else sorted_ratios[0]
        below_floor = [
            entry for entry in per_meeting if entry["fill_ratio"] < floor_target
        ]

        aggregate = {
            "duration_targeting_enabled": cfg.enabled,
            "meetings_checked": n,
            "mean_fill_ratio": round(mean, 4),
            "median_fill_ratio": round(median, 4),
            "p10_fill_ratio": round(p10, 4),
            "warn_below": round(warn_below, 4) if cfg.enabled else None,
            "floor_target": round(floor_target, 4) if cfg.enabled else None,
            "meetings_below_floor": len(below_floor),
            "below_floor_meeting_ids": [entry["meeting_id"] for entry in below_floor],
            "per_meeting": per_meeting,
        }
        try:
            fileio.write_json(self.output_dir / "duration_fill.json", aggregate)
        except Exception:
            logger.exception("Failed persisting duration_fill.json")

        results.append(ValidationResult(
            validator="meeting_duration_fill",
            severity=Severity.INFO,
            message=(
                f"Duration fill across {n} completed meetings: "
                f"mean {mean:.0%}, median {median:.0%}, p10 {p10:.0%} of scheduled slot"
                + (
                    f"; {len(below_floor)} below {floor_target:.0%} floor"
                    if cfg.enabled
                    else " (duration_targeting disabled)"
                )
            ),
            details={
                "meetings_checked": n,
                "mean_fill_ratio": round(mean, 4),
                "median_fill_ratio": round(median, 4),
                "p10_fill_ratio": round(p10, 4),
                "duration_targeting_enabled": cfg.enabled,
            },
        ))

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _save_report(self, results: list[ValidationResult]) -> None:
        # JSON
        fileio.write_json(
            self.output_dir / "validation_results.json",
            [r.to_dict() for r in results],
        )

        # Markdown
        errors = [r for r in results if r.severity == Severity.ERROR]
        warnings = [r for r in results if r.severity == Severity.WARNING]
        infos = [r for r in results if r.severity == Severity.INFO]

        lines = [
            "# Validation Report",
            "",
            f"**Errors:** {len(errors)} | **Warnings:** {len(warnings)} | **Info:** {len(infos)}",
            "",
        ]

        if errors:
            lines.append("## Errors")
            lines.append("")
            for r in errors:
                lines.append(f"- **[{r.validator}]** {r.message}")
            lines.append("")

        if warnings:
            lines.append("## Warnings")
            lines.append("")
            for r in warnings:
                lines.append(f"- **[{r.validator}]** {r.message}")
            lines.append("")

        if infos:
            lines.append("## Info")
            lines.append("")
            for r in infos:
                lines.append(f"- **[{r.validator}]** {r.message}")
            lines.append("")

        if not results:
            lines.append("No issues found.")

        path = self.output_dir / "validation_report.md"
        fileio.write_text(path, "\n".join(lines) + "\n")
        logger.info("Validation report saved to %s", path)
