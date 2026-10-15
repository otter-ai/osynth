"""Step 3: Decompose week plans into concrete artifact specs + dependency DAG.

Phases:
  1. Expand recurring meetings into calendar events
  2. Process week plans → artifact specs (meetings, emails, documents, slides)
  3. Build dependency graph and compute generation order (topological sort)
"""
from __future__ import annotations

import hashlib
import logging
import random
import uuid
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path

from ..models.calendar import (
    CalendarEvent,
    MeetingJoinMode,
    MeetingJoinSpec,
    ParticipantJoinInfo,
    ParticipantJoinMode,
    RecurringMeeting,
    Schedule,
    WorkArrangement,
)
from ..models.people import OrgStructure
from ..models.planning import (
    AdhocMeetingPlan,
    DocumentPlan,
    EmailThreadPlan,
    MeetingOutcomePlan,
    Narrative,
    SlidePresentationPlan,
    WeekPlan,
)
from ..models.settings import SyntheticDataSettings
from ..models.state import Project, WorkspaceState
from ..models.workspace import WorkspaceConfig
from ..utils import fileio
from ..utils.calendar_utils import CalendarScheduler
from ..utils.context_builder import ContextBuilder
from ..utils.id_validation import IdDrift, IdRegistry, dedupe_preserve_order

logger = logging.getLogger(__name__)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _is_unheld(meeting_spec: dict) -> bool:
    """True for meetings step 4 will not transcribe (planned skip/cancel)."""
    return meeting_spec.get("planned_outcome", "completed") in ("skipped", "canceled")


def _parse_preferred_time(
    raw: str | None,
    target_date: date,
    working_hours: tuple[int, int],
    *,
    default_hours: tuple[int, ...],
) -> datetime:
    """Combine ``target_date`` with an LLM-emitted ``preferred_time`` (HH:MM).

    Multi-tier fallback mirroring ``_create_adhoc_meeting``'s slot-finding
    chain: try the LLM's ``preferred_time`` first, then walk through a list
    of context-appropriate defaults (e.g. mid-morning, mid-afternoon,
    late-afternoon for documents), and finally clamp into ``working_hours``
    so nothing lands outside the configured workday.
    """
    lo, hi = working_hours

    if isinstance(raw, str) and raw:
        parts = raw.strip().split(":")
        try:
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            if 0 <= h <= 23 and 0 <= m <= 59:
                hour = max(lo, min(hi - 1, h))
                return datetime.combine(
                    target_date, datetime.min.time().replace(hour=hour, minute=m)
                )
        except (ValueError, IndexError):
            pass

    for default_hour in default_hours:
        if lo <= default_hour < hi:
            return datetime.combine(
                target_date, datetime.min.time().replace(hour=default_hour)
            )

    return datetime.combine(target_date, datetime.min.time().replace(hour=lo))


class Step3Decomposition:
    """Expand week plans into artifact specs and build dependency graph."""

    def __init__(self, settings: SyntheticDataSettings, output_root: str, data: dict) -> None:
        self.settings = settings
        self.data = data
        self.generate_slides: bool = settings.generation.generate_slides

        self.output_dir = Path(output_root) / "step_3_decomposition"

        self.workspace_config: WorkspaceConfig = data["workspace_config"]
        self.org: OrgStructure = data["org"]
        self.projects: dict[str, Project] = data["projects"]
        self.recurring_meetings: list[RecurringMeeting] = data["recurring_meetings"]
        self.run_seed: int = int(settings.seed) if settings.seed is not None else 0
        self.initial_state: WorkspaceState = data["initial_state"]
        self.narrative: Narrative = Narrative.model_validate(data["narrative"])
        self.week_plans: list[WeekPlan] = [
            plan if isinstance(plan, WeekPlan) else WeekPlan.model_validate(plan)
            for plan in data["week_plans"]
        ]
        self.ctx = ContextBuilder(self.workspace_config, self.org)

        self.scheduler = CalendarScheduler(
            working_hours=self.workspace_config.working_hours,
            working_days=self.workspace_config.working_days,
            timezone=self.workspace_config.timezone,
        )

        self.id_registry = IdRegistry.from_org(self.org, self.projects)
        self.id_drift = IdDrift()

        self.schedule = Schedule()
        self.meeting_specs: list[dict] = []
        self.email_specs: list[dict] = []
        self.document_specs: list[dict] = []
        self.slide_specs: list[dict] = []

    async def run(self, force: bool = False) -> dict:
        gen_order_path = self.output_dir / "generation_order.json"
        if gen_order_path.exists() and not force:
            logger.info("Step 3 outputs exist, loading from disk.")
            return self._load_from_disk()

        # Phase 1: Expand recurring meetings
        self._expand_recurring_meetings()

        # Phase 2: Process week plans
        self._process_week_plans()
        self._apply_meeting_plan()

        # Phase 3: Build dependency graph + generation order
        artifact_graph = self._build_artifact_graph()
        generation_order = self._topological_generation_order(artifact_graph)

        # Persist
        self._save_all(artifact_graph, generation_order)

        return {
            "schedule": self.schedule,
            "meeting_specs": self.meeting_specs,
            "email_specs": self.email_specs,
            "document_specs": self.document_specs,
            "slide_specs": self.slide_specs,
            "artifact_graph": artifact_graph,
            "generation_order": generation_order,
        }

    # ------------------------------------------------------------------
    # Phase 1: Recurring meetings → calendar events
    # ------------------------------------------------------------------

    def _expand_recurring_meetings(self) -> None:
        self.schedule.recurring_meetings = list(self.recurring_meetings)
        events = self.scheduler.expand_recurring_meetings(
            self.recurring_meetings,
            self.workspace_config.time_period_start,
            self.workspace_config.time_period_end,
        )
        for event in events:
            event.join_spec = self._sample_join_spec(
                event_id=event.id,
                attendee_ids=event.attendee_ids,
                missed_ids=set(),
            )
        self.schedule.events.extend(events)
        logger.info("Expanded %d recurring meetings into %d calendar events",
                     len(self.recurring_meetings), len(events))

    # ------------------------------------------------------------------
    # Phase 2: Week plans → artifact specs
    # ------------------------------------------------------------------

    def _process_week_plans(self) -> None:
        for plan in self.week_plans:
            week_start = date.fromisoformat(plan.week_start)

            # Map meeting outcomes by title for this week
            outcomes_by_title: dict[str, MeetingOutcomePlan] = {
                outcome.meeting_title: outcome for outcome in plan.meeting_outcomes
            }

            # Collect all events in this week
            week_event_list: list[CalendarEvent] = []
            for day_offset in range(7):
                d = week_start + timedelta(days=day_offset)
                week_event_list.extend(self.schedule.events_for_date(d))

            for event in week_event_list:
                if event.recurring_meeting_id:
                    outcome_data = outcomes_by_title.get(event.title)
                    # Re-derive join spec excluding missed attendees so the
                    # participant_modes map only covers people actually present.
                    missed_ids: set[str] = set()
                    if outcome_data:
                        for change in outcome_data.attendance_changes:
                            if (change.status or "").strip().lower() == "missed":
                                missed_ids.add(change.person_id)
                    event.join_spec = self._sample_join_spec(
                        event_id=event.id,
                        attendee_ids=event.attendee_ids,
                        missed_ids=missed_ids,
                    )
                    spec = self._create_meeting_spec(event, outcome_data)
                    self.meeting_specs.append(spec)

            # Adhoc meetings
            for adhoc in plan.adhoc_meetings:
                event, spec = self._create_adhoc_meeting(adhoc, week_start)
                if event:
                    self.schedule.events.append(event)
                    self.meeting_specs.append(spec)

            # Emails
            for email_plan in plan.email_threads:
                spec = self._create_email_spec(email_plan, week_start)
                self.email_specs.append(spec)

            # Documents
            for doc_plan in plan.documents:
                spec = self._create_document_spec(doc_plan, week_start)
                self.document_specs.append(spec)

            # Slides
            if self.generate_slides:
                for slide_plan in plan.slide_presentations:
                    spec = self._create_slide_spec(slide_plan, week_start)
                    self.slide_specs.append(spec)

        logger.info(
            "Created specs: %d meetings, %d emails, %d documents, %d slides",
            len(self.meeting_specs), len(self.email_specs),
            len(self.document_specs), len(self.slide_specs),
        )

    # ------------------------------------------------------------------
    # Phase 2b: Meeting quota / slot override (generation.meeting_plan)
    # ------------------------------------------------------------------

    def _apply_meeting_plan(self) -> None:
        plan = self.settings.generation.meeting_plan
        if plan.count is None and plan.duration_minutes is None:
            return

        # Slot override first, so the truncation cutoff sees the final end times.
        if plan.duration_minutes is not None:
            events_by_id = {e.id: e for e in self.schedule.events}
            for spec in self.meeting_specs:
                start = datetime.fromisoformat(spec["start_time"])
                end = start + timedelta(minutes=plan.duration_minutes)
                spec["end_time"] = end.isoformat()
                spec["duration_minutes"] = plan.duration_minutes
                event = events_by_id.get(spec["calendar_event_id"])
                if event is not None:
                    event.end_time = end

        if plan.count is not None:
            meetings = sorted(self.meeting_specs, key=lambda s: s["start_time"])
            # Only meetings that will actually be held count toward the quota:
            # step 4 writes no transcript for planned skips/cancellations.
            # Those before the cutoff stay on the calendar as context.
            held = [i for i, s in enumerate(meetings) if not _is_unheld(s)]
            if len(held) < plan.count:
                raise ValueError(
                    f"meeting_plan.count={plan.count} but the planner produced only "
                    f"{len(held)} held meetings ({len(meetings)} total); widen the "
                    "workspace time period."
                )
            last_kept = held[plan.count - 1]
            kept, dropped = meetings[: last_kept + 1], meetings[last_kept + 1:]
            if dropped:
                self._truncate_after(kept, dropped)
            self.meeting_specs = kept

        logger.info(
            "Applied meeting_plan (count=%s, duration_minutes=%s): %d meetings, "
            "%d emails, %d documents, %d slides",
            plan.count, plan.duration_minutes, len(self.meeting_specs),
            len(self.email_specs), len(self.document_specs), len(self.slide_specs),
        )

    def _truncate_after(self, kept: list[dict], dropped: list[dict]) -> None:
        """End the period at the last kept meeting.

        Drops the surplus meetings (and their calendar events), follow-up
        emails of dropped meetings, and any other artifact dated after the
        last kept meeting ends. Earlier documents that merely referenced a
        dropped meeting are kept but unlinked.
        """
        cutoff = max(datetime.fromisoformat(s["end_time"]) for s in kept)
        dropped_event_ids = {s["calendar_event_id"] for s in dropped}
        dropped_titles = {s["title"] for s in dropped} - {s["title"] for s in kept}

        def before_cutoff(spec: dict) -> bool:
            return datetime.fromisoformat(spec["timestamp"]) <= cutoff

        n_before = (len(self.email_specs), len(self.document_specs), len(self.slide_specs))
        self.email_specs = [
            s for s in self.email_specs
            if before_cutoff(s)
            and not (s.get("is_followup") and s.get("related_meeting_id") in dropped_event_ids)
        ]
        self.document_specs = [s for s in self.document_specs if before_cutoff(s)]
        for spec in self.email_specs + self.document_specs:
            if spec.get("related_meeting_id") in dropped_event_ids:
                spec["related_meeting_id"] = None
        self.slide_specs = [
            s for s in self.slide_specs
            if before_cutoff(s) and s.get("meeting_title") not in dropped_titles
        ]
        self.schedule.events = [e for e in self.schedule.events if e.id not in dropped_event_ids]

        logger.info(
            "meeting_plan: kept %d meetings, dropped %d (period ends %s); dropped "
            "%d emails, %d documents, %d slides",
            len(kept), len(dropped), cutoff.isoformat(),
            n_before[0] - len(self.email_specs), n_before[1] - len(self.document_specs),
            n_before[2] - len(self.slide_specs),
        )

    # ------------------------------------------------------------------
    # Deterministic meeting join-mode sampling
    # ------------------------------------------------------------------

    def _sample_join_spec(
        self,
        *,
        event_id: str,
        attendee_ids: list[str],
        missed_ids: set[str],
        llm_override: MeetingJoinSpec | None = None,
    ) -> MeetingJoinSpec:
        """Sample meeting modality + per-attendee join modes deterministically.

        Seed comes from ``(run_seed, event_id)`` so reruns reproduce, and
        each recurring-meeting occurrence varies independently across the
        time period. Participant modes always cover only the currently
        active attendees (no entries for ``missed_ids``).
        """
        active_ids = [pid for pid in attendee_ids if pid not in missed_ids]

        seed_input = f"{self.run_seed}:{event_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha1(seed_input).digest()[:8], "big")
        rng = random.Random(seed)

        if llm_override is not None:
            mode = llm_override.mode
            room_name = llm_override.room_name
            notes = llm_override.notes
        else:
            mode = self._pick_meeting_mode(rng)
            room_name = ""
            notes = ""

        participant_modes = self._assign_participant_modes(
            mode=mode, active_ids=active_ids, rng=rng,
        )

        if mode in (MeetingJoinMode.IN_PERSON, MeetingJoinMode.HYBRID) and not room_name:
            room_name = self._pick_room_name(rng)

        return MeetingJoinSpec(
            mode=mode,
            participant_modes=participant_modes,
            room_name=room_name,
            notes=notes,
        )

    def _pick_meeting_mode(self, rng: random.Random) -> MeetingJoinMode:
        dist = self.settings.generation.meeting_join_mode_distribution
        items: list[tuple[MeetingJoinMode, float]] = []
        for k, v in dist.items():
            try:
                items.append((MeetingJoinMode(k), max(0.0, float(v))))
            except ValueError:
                continue
        total = sum(w for _, w in items)
        if total <= 0 or not items:
            return MeetingJoinMode.VIRTUAL
        r = rng.random() * total
        cum = 0.0
        for mode, w in items:
            cum += w
            if r <= cum:
                return mode
        return items[-1][0]

    def _fallback_work_arrangement(self, rng: random.Random) -> WorkArrangement:
        dist = self.settings.generation.work_arrangement_distribution
        items: list[tuple[WorkArrangement, float]] = []
        for k, v in dist.items():
            try:
                items.append((WorkArrangement(k), max(0.0, float(v))))
            except ValueError:
                continue
        total = sum(w for _, w in items)
        if total <= 0 or not items:
            return WorkArrangement.HYBRID
        r = rng.random() * total
        cum = 0.0
        for arr, w in items:
            cum += w
            if r <= cum:
                return arr
        return items[-1][0]

    def _assign_participant_modes(
        self,
        *,
        mode: MeetingJoinMode,
        active_ids: list[str],
        rng: random.Random,
    ) -> dict[str, ParticipantJoinInfo]:
        if mode == MeetingJoinMode.VIRTUAL:
            return {
                pid: ParticipantJoinInfo(mode=ParticipantJoinMode.REMOTE)
                for pid in active_ids
            }

        if mode == MeetingJoinMode.IN_PERSON:
            modes: dict[str, ParticipantJoinInfo] = {}
            for pid in active_ids:
                # External attendees can't realistically be in-room at the
                # host's office, so they dial in by phone.
                if not self.org.is_internal(pid):
                    modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.PHONE)
                else:
                    modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.IN_ROOM)
            return modes

        # HYBRID
        modes = {}
        in_room_ids: list[str] = []
        remote_ids: list[str] = []
        for pid in active_ids:
            person = self.org.get_person(pid)
            is_external = (person is not None) and not self.org.is_internal(pid)
            if is_external:
                modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.REMOTE)
                remote_ids.append(pid)
                continue
            arrangement: WorkArrangement | None = getattr(
                person, "work_arrangement", None
            ) if person else None
            if arrangement is None:
                arrangement = self._fallback_work_arrangement(rng)
            if arrangement == WorkArrangement.REMOTE:
                modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.REMOTE)
                remote_ids.append(pid)
            elif arrangement == WorkArrangement.IN_PERSON:
                modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.IN_ROOM)
                in_room_ids.append(pid)
            else:
                if rng.random() < 0.5:
                    modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.IN_ROOM)
                    in_room_ids.append(pid)
                else:
                    modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.REMOTE)
                    remote_ids.append(pid)

        # Guarantee at least one of each side when 2+ active attendees.
        if len(active_ids) >= 2:
            if not in_room_ids and remote_ids:
                candidates = [
                    pid for pid in remote_ids
                    if self.org.is_internal(pid)
                    and getattr(self.org.get_person(pid), "work_arrangement", None)
                    != WorkArrangement.REMOTE
                ] or [pid for pid in remote_ids if self.org.is_internal(pid)]
                if candidates:
                    pid = rng.choice(candidates)
                    modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.IN_ROOM)
                    remote_ids.remove(pid)
                    in_room_ids.append(pid)
            if not remote_ids and in_room_ids:
                candidates = [
                    pid for pid in in_room_ids
                    if getattr(self.org.get_person(pid), "work_arrangement", None)
                    != WorkArrangement.IN_PERSON
                ] or in_room_ids
                pid = rng.choice(candidates)
                modes[pid] = ParticipantJoinInfo(mode=ParticipantJoinMode.REMOTE)
                in_room_ids.remove(pid)
                remote_ids.append(pid)

        return modes

    def _pick_room_name(self, rng: random.Random) -> str:
        names = [
            "Main Conference Room",
            "Boardroom",
            "Huddle Room A",
            "Huddle Room B",
            "Conference Room 1",
            "Conference Room 2",
        ]
        return rng.choice(names)

    def _create_meeting_spec(
        self, event: CalendarEvent, outcome_data: MeetingOutcomePlan | None
    ) -> dict:
        planned_outcome = outcome_data.outcome if outcome_data else "completed"
        attendance_changes = {}
        if outcome_data:
            for change in outcome_data.attendance_changes:
                attendance_changes[change.person_id] = change.status or "attended"

        attendee_info = []
        for pid in event.attendee_ids:
            person = self.org.get_person(pid)
            if person:
                attendee_info.append({
                    "person_id": pid,
                    "name": person.name,
                    "gender": person.gender,
                    "is_external": not self.org.is_internal(pid),
                })

        return {
            "id": _uid("meeting"),
            "type": "meeting",
            "calendar_event_id": event.id,
            "meeting_type": event.meeting_type,
            "title": event.title,
            "start_time": event.start_time.isoformat(),
            "end_time": event.end_time.isoformat(),
            "duration_minutes": int((event.end_time - event.start_time).total_seconds() / 60),
            "organizer_id": event.organizer_id,
            "attendees": attendee_info,
            "attendee_ids": list(event.attendee_ids),
            "purpose": event.description,
            "agenda": list(event.agenda),
            "labels": event.labels.model_dump(mode="json") if event.labels else {},
            "join_spec": event.join_spec.model_dump(mode="json"),
            "related_project_id": None,
            "planned_outcome": planned_outcome,
            "planned_attendance_changes": attendance_changes,
            "should_generate_slides": False,
            "related_slide_id": None,
            "related_document_ids": [],
            "timestamp": event.start_time.isoformat(),
            "depends_on": [],
            "related_artifact_ids": [],
        }

    def _create_adhoc_meeting(
        self, adhoc: AdhocMeetingPlan, week_start: date
    ) -> tuple[CalendarEvent | None, dict]:
        day_of_week = adhoc.day_of_week
        target_date = week_start + timedelta(days=day_of_week)
        # Clamp to a valid working day
        target_date = self.scheduler.clamp_to_working_day(target_date)
        duration = adhoc.duration_minutes

        # Validate attendee IDs against org
        title = adhoc.title or "adhoc meeting"
        ctx = f"adhoc:{title}"
        attendee_ids = self.id_drift.filter_person_ids(
            adhoc.attendee_ids,
            self.id_registry,
            field_name="attendee_ids",
            context=ctx,
        )

        slot = self.scheduler.find_available_slot(
            attendee_ids,
            target_date,
            duration,
            self.schedule,
            preferred_time=adhoc.preferred_time,
        )
        if slot is None:
            # Try other working days in the week
            slot = self.scheduler.find_available_slot_in_week(
                attendee_ids,
                week_start,
                duration,
                self.schedule,
            )
        if slot is None:
            logger.warning("No slot found for adhoc meeting: %s", adhoc.title)
            # Force-place on a working day
            fallback_date = self.scheduler.clamp_to_working_day(target_date)
            slot = datetime.combine(fallback_date, datetime.min.time().replace(hour=10))

        end_time = slot + timedelta(minutes=duration)

        labels = adhoc.labels

        event_id = _uid("cal")
        # Validate organizer
        organizer_id = self.id_drift.resolve_person_id(
            adhoc.organizer_id or "",
            self.id_registry,
            field_name="organizer_id",
            context=ctx,
            fallback=attendee_ids[0] if attendee_ids else None,
        )
        related_project_id = self.id_drift.normalize_project_id(
            adhoc.related_project_id,
            self.id_registry,
            field_name="related_project_id",
            context=ctx,
        )

        join_spec = self._sample_join_spec(
            event_id=event_id,
            attendee_ids=attendee_ids,
            missed_ids=set(),
            llm_override=adhoc.join_spec,
        )

        event = CalendarEvent(
            id=event_id,
            title=adhoc.title or "Adhoc Meeting",
            meeting_type=adhoc.meeting_type or "adhoc",
            start_time=slot,
            end_time=end_time,
            organizer_id=organizer_id,
            attendee_ids=attendee_ids,
            description=adhoc.purpose,
            labels=labels,
            join_spec=join_spec,
        )

        attendee_info = []
        for pid in event.attendee_ids:
            person = self.org.get_person(pid)
            if person:
                attendee_info.append({
                    "person_id": pid,
                    "name": person.name,
                    "gender": person.gender,
                    "is_external": not self.org.is_internal(pid),
                })

        spec = {
            "id": _uid("meeting"),
            "type": "meeting",
            "calendar_event_id": event_id,
            "meeting_type": event.meeting_type,
            "title": event.title,
            "start_time": slot.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration,
            "organizer_id": event.organizer_id,
            "attendees": attendee_info,
            "attendee_ids": list(event.attendee_ids),
            "purpose": adhoc.purpose,
            "agenda": [],
            "expected_outcomes": list(adhoc.expected_outcomes),
            "labels": labels.model_dump(mode="json"),
            "join_spec": event.join_spec.model_dump(mode="json"),
            "related_project_id": related_project_id,
            "planned_outcome": "completed",
            "planned_attendance_changes": {},
            "should_generate_slides": bool(adhoc.should_generate_slides and self.generate_slides),
            "related_slide_id": None,
            "related_document_ids": [],
            "timestamp": slot.isoformat(),
            "depends_on": [],
            "related_artifact_ids": [],
        }

        return event, spec

    def _create_email_spec(self, email_plan: EmailThreadPlan, week_start: date) -> dict:
        day_of_week = email_plan.day_of_week
        target_date = week_start + timedelta(days=day_of_week)
        target_date = self.scheduler.clamp_to_working_day(target_date)
        # Email defaults: lean toward mid-morning for routine threads, with
        # late-morning and early-afternoon as next choices. Urgent threads
        # are handled by the LLM emitting an explicit early ``preferred_time``.
        start_time = _parse_preferred_time(
            email_plan.preferred_time,
            target_date,
            self.workspace_config.working_hours,
            default_hours=(10, 11, 14, 9),
        )

        # Validate person IDs
        subject = email_plan.subject
        ctx = f"email:{subject}"
        participant_ids = self.id_drift.filter_person_ids(
            email_plan.participant_ids,
            self.id_registry,
            field_name="participant_ids",
            context=ctx,
        )
        participant_ids = dedupe_preserve_order(participant_ids)
        initiator_id = self.id_drift.resolve_person_id(
            email_plan.initiator_id or "",
            self.id_registry,
            field_name="initiator_id",
            context=ctx,
            fallback=participant_ids[0] if participant_ids else None,
        )
        if initiator_id and initiator_id not in participant_ids:
            participant_ids = [initiator_id, *participant_ids]
            self.id_drift.record(
                kind="person",
                field_name="participant_ids",
                context=ctx,
                bad=initiator_id,
                action="prepended_initiator",
            )
        cc_ids = dedupe_preserve_order(self.id_drift.filter_person_ids(
            email_plan.cc_ids,
            self.id_registry,
            field_name="cc_ids",
            context=ctx,
        ))
        overlapping_cc_ids = [pid for pid in cc_ids if pid in participant_ids]
        if overlapping_cc_ids:
            self.id_drift.record(
                kind="person",
                field_name="cc_ids",
                context=ctx,
                bad=overlapping_cc_ids,
                action="dropped_overlap_with_participant_ids",
            )
            cc_ids = [pid for pid in cc_ids if pid not in participant_ids]
        related_project_id = self.id_drift.normalize_project_id(
            email_plan.related_project_id,
            self.id_registry,
            field_name="related_project_id",
            context=ctx,
        )

        return {
            "id": _uid("thread"),
            "type": "email_thread",
            "subject": subject,
            "initiator_id": initiator_id,
            "participant_ids": participant_ids,
            "cc_ids": cc_ids,
            "start_time": start_time.isoformat(),
            "num_messages": email_plan.num_messages,
            "purpose": email_plan.purpose,
            "tone": email_plan.tone or "professional",
            "related_meeting_id": email_plan.related_meeting_id,
            "related_project_id": related_project_id,
            "is_followup": email_plan.is_followup,
            "timestamp": start_time.isoformat(),
            "depends_on": [],
            "related_artifact_ids": [],
        }

    def _create_document_spec(self, doc_plan: DocumentPlan, week_start: date) -> dict:
        day_of_week = doc_plan.day_of_week
        target_date = week_start + timedelta(days=day_of_week)
        target_date = self.scheduler.clamp_to_working_day(target_date)
        # Document defaults: weekly status, reports, and EOD recaps tend to
        # land late afternoon; specs and proposals more often mid-afternoon
        # or late-morning. Order reflects that preference.
        created_at = _parse_preferred_time(
            doc_plan.preferred_time,
            target_date,
            self.workspace_config.working_hours,
            default_hours=(16, 14, 11, 15),
        )

        # Validate author
        title = doc_plan.title
        ctx = f"doc:{title}"
        author_id = self.id_drift.resolve_person_id(
            doc_plan.author_id or "",
            self.id_registry,
            field_name="author_id",
            context=ctx,
        )
        related_project_id = self.id_drift.normalize_project_id(
            doc_plan.related_project_id,
            self.id_registry,
            field_name="related_project_id",
            context=ctx,
        )

        return {
            "id": _uid("doc"),
            "type": "document",
            "title": title,
            "author_id": author_id,
            "doc_type": doc_plan.doc_type,
            "created_at": created_at.isoformat(),
            "purpose": doc_plan.purpose,
            "related_project_id": related_project_id,
            "related_meeting_id": doc_plan.related_meeting_id,
            "approximate_length": doc_plan.approximate_length,
            "timestamp": created_at.isoformat(),
            "depends_on": [],
            "related_artifact_ids": [],
        }

    def _create_slide_spec(
        self, slide_plan: SlidePresentationPlan, week_start: date
    ) -> dict:
        # Slides are created before the meeting they belong to
        title = slide_plan.title
        author_id = self.id_drift.resolve_person_id(
            slide_plan.author_id or "",
            self.id_registry,
            field_name="author_id",
            context=f"slides:{title}",
        )

        return {
            "id": _uid("slides"),
            "type": "slides",
            "title": title,
            "author_id": author_id,
            "meeting_title": slide_plan.meeting_title,
            "num_slides": slide_plan.num_slides,
            "purpose": slide_plan.purpose,
            "key_points": list(slide_plan.key_points),
            "timestamp": week_start.isoformat(),
            "depends_on": [],
            "related_artifact_ids": [],
        }

    # ------------------------------------------------------------------
    # Phase 3: Dependency graph
    # ------------------------------------------------------------------

    def _build_artifact_graph(self) -> dict:
        all_specs: list[dict] = (
            self.meeting_specs + self.email_specs + self.document_specs + self.slide_specs
        )
        edges: list[dict] = []

        # Index meetings by title for matching
        meetings_by_title: dict[str, list[dict]] = defaultdict(list)
        for spec in self.meeting_specs:
            meetings_by_title[spec["title"]].append(spec)

        # Slides → Meeting dependencies (slides must be generated before the meeting)
        for slide_spec in self.slide_specs:
            meeting_title = slide_spec.get("meeting_title")
            if not meeting_title:
                continue
            matched_meetings = meetings_by_title.get(meeting_title, [])
            for meeting_spec in matched_meetings:
                meeting_spec["depends_on"].append(slide_spec["id"])
                meeting_spec["should_generate_slides"] = True
                meeting_spec["related_slide_id"] = slide_spec["id"]
                slide_spec["related_artifact_ids"].append(meeting_spec["id"])
                edges.append({
                    "from": slide_spec["id"],
                    "to": meeting_spec["id"],
                    "kind": "meeting_input",
                })

        # Documents → Meeting dependencies (docs referenced by a meeting)
        for doc_spec in self.document_specs:
            related_meeting_id = doc_spec.get("related_meeting_id")
            if not related_meeting_id:
                continue
            # Find meeting spec by calendar event pattern
            for meeting_spec in self.meeting_specs:
                if meeting_spec.get("calendar_event_id") == related_meeting_id:
                    meeting_spec["depends_on"].append(doc_spec["id"])
                    meeting_spec["related_document_ids"].append(doc_spec["id"])
                    doc_spec["related_artifact_ids"].append(meeting_spec["id"])
                    edges.append({
                        "from": doc_spec["id"],
                        "to": meeting_spec["id"],
                        "kind": "meeting_input",
                    })

        # Follow-up emails depend on their source meeting
        for email_spec in self.email_specs:
            if not email_spec.get("is_followup"):
                continue
            related_meeting_id = email_spec.get("related_meeting_id")
            if not related_meeting_id:
                continue
            for meeting_spec in self.meeting_specs:
                if meeting_spec.get("calendar_event_id") == related_meeting_id:
                    email_spec["depends_on"].append(meeting_spec["id"])
                    email_spec["related_artifact_ids"].append(meeting_spec["id"])
                    edges.append({
                        "from": meeting_spec["id"],
                        "to": email_spec["id"],
                        "kind": "followup_email",
                    })

        # Build nodes
        nodes = []
        for spec in all_specs:
            nodes.append({
                "id": spec["id"],
                "type": spec["type"],
                "timestamp": spec.get("timestamp", ""),
                "depends_on": spec.get("depends_on", []),
                "related_artifact_ids": spec.get("related_artifact_ids", []),
            })

        logger.info("Built artifact graph: %d nodes, %d edges", len(nodes), len(edges))
        return {"nodes": nodes, "edges": edges}

    def _topological_generation_order(self, graph: dict) -> list[dict]:
        """Kahn's algorithm for topological sort, with timestamp tiebreaking."""
        nodes = {n["id"]: n for n in graph["nodes"]}
        in_degree: dict[str, int] = {nid: 0 for nid in nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in nodes}

        for edge in graph["edges"]:
            src, dst = edge["from"], edge["to"]
            if src in adj and dst in in_degree:
                adj[src].append(dst)
                in_degree[dst] += 1

        # Start with nodes that have no dependencies
        queue: deque[str] = deque()
        for nid, deg in sorted(in_degree.items(), key=lambda x: (nodes[x[0]].get("timestamp", ""), x[0])):
            if deg == 0:
                queue.append(nid)

        order: list[dict] = []
        while queue:
            # Sort current batch by timestamp for determinism
            batch = sorted(queue, key=lambda nid: (nodes[nid].get("timestamp", ""), nodes[nid].get("type", ""), nid))
            queue.clear()

            for nid in batch:
                node = nodes[nid]
                order.append({
                    "id": nid,
                    "type": node["type"],
                    "timestamp": node.get("timestamp", ""),
                    "depends_on": node.get("depends_on", []),
                })
                for neighbor in adj[nid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # If cycle detected, add remaining by timestamp
        if len(order) < len(nodes):
            remaining = set(nodes) - {o["id"] for o in order}
            logger.warning("Cycle detected in artifact graph; %d nodes added by timestamp fallback", len(remaining))
            for nid in sorted(remaining, key=lambda x: (nodes[x].get("timestamp", ""), x)):
                node = nodes[nid]
                order.append({
                    "id": nid,
                    "type": node["type"],
                    "timestamp": node.get("timestamp", ""),
                    "depends_on": node.get("depends_on", []),
                })

        logger.info("Generation order: %d artifacts", len(order))
        return order

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_all(self, artifact_graph: dict, generation_order: list[dict]) -> None:
        fileio.makedirs(self.output_dir)
        fileio.write_json(self.output_dir / "calendar.json", self.schedule.model_dump(mode="json"))
        fileio.write_json(self.output_dir / "artifact_graph.json", artifact_graph)
        fileio.write_json(self.output_dir / "generation_order.json", generation_order)
        self.id_drift.persist(self.output_dir / "id_drift.json")

        # Save specs by type
        for spec_type, specs, subdir in [
            ("meetings", self.meeting_specs, "meetings"),
            ("emails", self.email_specs, "emails"),
            ("documents", self.document_specs, "documents"),
            ("slides", self.slide_specs, "slides"),
        ]:
            spec_dir = self.output_dir / "artifact_specs" / subdir
            fileio.makedirs(spec_dir)
            for spec in specs:
                fileio.write_json(spec_dir / f"{spec['id']}.json", spec)

        logger.info("Step 3 outputs saved to %s", self.output_dir)

    def _load_from_disk(self) -> dict:
        schedule = Schedule.model_validate(fileio.read_json(self.output_dir / "calendar.json"))
        artifact_graph = fileio.read_json(self.output_dir / "artifact_graph.json")
        generation_order = fileio.read_json(self.output_dir / "generation_order.json")

        specs: dict[str, list[dict]] = {
            "meeting_specs": [], "email_specs": [], "document_specs": [], "slide_specs": [],
        }
        type_map = {
            "meetings": "meeting_specs", "emails": "email_specs",
            "documents": "document_specs", "slides": "slide_specs",
        }
        for subdir, key in type_map.items():
            spec_dir = self.output_dir / "artifact_specs" / subdir
            if spec_dir.exists():
                for fname in sorted(spec_dir.iterdir()):
                    if fname.suffix == ".json":
                        specs[key].append(fileio.read_json(fname))

        logger.info("Loaded step 3: %d meeting, %d email, %d doc, %d slide specs",
                     len(specs["meeting_specs"]), len(specs["email_specs"]),
                     len(specs["document_specs"]), len(specs["slide_specs"]))

        return {
            "schedule": schedule,
            **specs,
            "artifact_graph": artifact_graph,
            "generation_order": generation_order,
        }
