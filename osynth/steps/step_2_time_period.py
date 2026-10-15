"""Step 2: Time period planning — narrative arc + per-week plans.

Two-level hierarchy:
  1. Full-period narrative (1 LLM call)
  2. Per-week plans built via authoritative-frame-plus-fanout:
     2a. ``week_frame``    — shared scope, summary, state changes, content targets
     2b. parallel family fanout — meetings, emails, documents
     2c. ``week_slides``   — depends on the meetings family
     2d. typed assembly into a single ``WeekPlan`` artifact
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ..models.calendar import RecurringMeeting
from ..models.people import OrgStructure
from ..models.planning import (
    Narrative,
    WeekDocumentsOutput,
    WeekEmailsOutput,
    WeekFrame,
    WeekMeetingsOutput,
    WeekPlan,
    WeekSlidesOutput,
    validate_project_arcs,
)
from ..models.settings import SyntheticDataSettings
from ..models.state import (
    EventLog,
    EventType,
    Project,
    ProjectStatus,
    WorkspaceEvent,
    WorkspaceState,
)
from ..models.workspace import WorkspaceConfig
from ..utils import fileio
from ..utils.context_builder import ContextBuilder
from ..utils.id_validation import IdDrift, IdRegistry, sanitize_week_plan
from ..utils.llm_json import extract_json
from ..utils.llm_client import LLMClient, LLMResponse
from ..utils.prompt import read_prompt, render_prompt
from ..utils.prompt_trace import PromptTraceRecorder

logger = logging.getLogger(__name__)


def _num_weeks(start: date, end: date) -> int:
    """Number of (possibly partial) weeks in the inclusive range ``start``..``end``."""
    return max(1, -(-((end - start).days + 1) // 7))

MAX_RETRIES = 2


class Step2TimePeriod:
    """Generate narrative arc and per-week plans."""

    def __init__(
        self,
        settings: SyntheticDataSettings,
        output_root: str,
        data: dict,
        *,
        llm_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.data = data

        self.model: str = settings.model
        self.llm = llm_client or LLMClient(model=self.model)
        self.generate_slides: bool = settings.generation.generate_slides

        self.output_dir = Path(output_root) / "step_2_time_period"
        self.prompt_tracer = PromptTraceRecorder(self.output_dir / "prompt_traces")

        self.language: str = settings.language
        code_root = Path(__file__).resolve().parent.parent
        prompts_cfg = settings.prompts.time_period
        self.prompts = {
            "system": read_prompt(code_root / prompts_cfg.system, self.language),
            "narrative": read_prompt(code_root / prompts_cfg.narrative, self.language),
            "week_frame": read_prompt(code_root / prompts_cfg.week_frame, self.language),
            "week_meetings": read_prompt(code_root / prompts_cfg.week_meetings, self.language),
            "week_emails": read_prompt(code_root / prompts_cfg.week_emails, self.language),
            "week_documents": read_prompt(code_root / prompts_cfg.week_documents, self.language),
            "week_slides": read_prompt(code_root / prompts_cfg.week_slides, self.language),
        }

        # Per-call output-token budgets. Week plans are the most truncation-prone
        # call in the pipeline because each must contain meetings, emails,
        # documents, slides, and state changes for the whole week.
        self.budget = settings.generation.llm.step_2

        self.workspace_config: WorkspaceConfig = data["workspace_config"]
        self.org: OrgStructure = data["org"]
        self.projects: dict[str, Project] = data["projects"]
        self.recurring_meetings: list[RecurringMeeting] = data["recurring_meetings"]
        self.initial_state: WorkspaceState = data["initial_state"]
        self.ctx = ContextBuilder(self.workspace_config, self.org)
        self.id_drift = IdDrift()
        self.planning_log = EventLog()

    async def run(self, force: bool = False) -> dict:
        narrative_path = self.output_dir / "narrative.json"
        if narrative_path.exists() and not force:
            logger.info("Step 2 outputs exist, loading from disk.")
            return self._load_from_disk()

        # Phase 1: Narrative arc
        narrative = await self._generate_narrative()

        # Phase 2: Week-by-week plans (sequential)
        week_plans = await self._generate_week_plans(narrative)

        # Extract project trajectories
        project_trajectories = narrative.get("project_arcs", [])

        # Persist
        self._save_all(narrative, week_plans, project_trajectories)

        return {
            "narrative": narrative,
            "week_plans": week_plans,
            "project_trajectories": project_trajectories,
        }

    # ------------------------------------------------------------------
    # LLM API
    # ------------------------------------------------------------------

    async def _call_llm(
        self, user_prompt: str, *, max_tokens: int = 8192, trace_label: str = "call",
    ) -> LLMResponse:
        from ..utils.usage_tracker import tracker

        response = await self.llm.create_message(
            system=self.prompts["system"],
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=1.0,
        )
        tracker.track_llm_response(response)
        self.prompt_tracer.record(
            label=trace_label,
            system_prompt=self.prompts["system"],
            user_prompt=user_prompt,
            response_text=response.text,
            metadata={
                "model": self.model,
                "max_tokens": max_tokens,
                "stop_reason": response.stop_reason,
                "output_tokens": response.output_tokens,
            },
        )
        return response

    async def _call_llm_json(
        self, user_prompt: str, *, max_tokens: int = 8192, trace_label: str = "json",
        expect_dict: bool = False,
    ) -> Any:
        last_error: Exception | None = None
        last_response: LLMResponse | None = None
        for attempt in range(1 + MAX_RETRIES):
            if attempt > 0:
                logger.warning("Retry %d/%d", attempt, MAX_RETRIES)
            response = await self._call_llm(
                user_prompt, max_tokens=max_tokens, trace_label=f"{trace_label}_attempt_{attempt + 1}",
            )
            last_response = response
            try:
                parsed = extract_json(response.text)
                if expect_dict and not isinstance(parsed, dict):
                    # Malformed objects (e.g. a bare string where a key belongs)
                    # make extract_json fall back to an inner fragment such as
                    # the ``themes`` list; retry instead of validating that.
                    raise ValueError(
                        f"expected a JSON object, got {type(parsed).__name__}"
                    )
                return parsed
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if response.truncated:
                    logger.warning(
                        "JSON parse failed for %s (attempt %d): output truncated by provider "
                        "(stop_reason=%s, output_tokens=%d, max_tokens=%d). "
                        "Raise the per-call max_tokens budget or decompose the request.",
                        trace_label, attempt + 1, response.stop_reason,
                        response.output_tokens, max_tokens,
                    )
                else:
                    logger.warning(
                        "JSON parse failed for %s (attempt %d, stop_reason=%s): %s",
                        trace_label, attempt + 1, response.stop_reason, exc,
                    )
        truncation_note = ""
        if last_response is not None and last_response.truncated:
            truncation_note = (
                f" (last response truncated by provider; stop_reason="
                f"{last_response.stop_reason}, max_tokens={max_tokens})"
            )
        raise ValueError(
            f"Failed to parse JSON from LLM for {trace_label}{truncation_note}"
        ) from last_error

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------

    async def _generate_narrative(self) -> dict:
        start = self.workspace_config.time_period_start
        end = self.workspace_config.time_period_end
        num_weeks = _num_weeks(start, end)

        company_json = self.workspace_config.company.model_dump_json(indent=2)
        org_summary = self.ctx.org_summary()
        projects_json = json.dumps(
            [p.model_dump(mode="json") for p in self.projects.values()], indent=2, default=str,
        )

        user_prompt = (
            self.prompts["narrative"]
            .replace("{{COMPANY_JSON}}", company_json)
            .replace("{{ORG_SUMMARY}}", org_summary)
            .replace("{{PROJECTS_JSON}}", projects_json)
            .replace("{{START_DATE}}", start.isoformat())
            .replace("{{END_DATE}}", end.isoformat())
            .replace("{{NUM_WEEKS}}", str(num_weeks))
        )

        logger.info("Generating narrative arc for %d weeks (%s to %s)", num_weeks, start, end)
        narrative = Narrative.model_validate(
            await self._call_llm_json(
                user_prompt,
                max_tokens=self.budget.narrative,
                trace_label="narrative",
                expect_dict=True,
            )
        )
        return narrative.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Week plans
    # ------------------------------------------------------------------

    async def _generate_week_plans(self, narrative: dict) -> list[dict]:
        start = self.workspace_config.time_period_start
        end = self.workspace_config.time_period_end
        num_weeks = _num_weeks(start, end)

        company_summary = self.ctx.company_summary()
        narrative_summary = narrative.get("narrative_summary", "")
        major_events = narrative.get("major_events", [])

        registry = IdRegistry.from_org(self.org, self.projects)
        self.id_drift = IdDrift()
        self.planning_log = EventLog()

        week_plans: list[dict] = []
        prev_forward_brief = "First week of simulation — no prior context."

        for week_num in range(1, num_weeks + 1):
            week_start = start + timedelta(weeks=week_num - 1)
            week_end = min(week_start + timedelta(days=6), end)

            current_state = self.planning_log.replay(self.initial_state)

            common_ctx = self._build_week_common_context(
                week_num=week_num,
                num_weeks=num_weeks,
                week_start=week_start,
                week_end=week_end,
                company_summary=company_summary,
                narrative_summary=narrative_summary,
                major_events=major_events,
                current_state=current_state,
                prev_forward_brief=prev_forward_brief,
            )

            logger.info(
                "Generating week %d/%d frame (%s to %s)",
                week_num, num_weeks, week_start, week_end,
            )
            frame = await self._generate_week_frame(common_ctx)
            frame_json = json.dumps(frame.model_dump(mode="json"), indent=2)

            logger.info(
                "Generating week %d/%d family fanout (meetings, emails, documents)",
                week_num, num_weeks,
            )
            meetings, emails, documents = await asyncio.gather(
                self._generate_week_meetings(common_ctx, frame_json),
                self._generate_week_emails(common_ctx, frame_json),
                self._generate_week_documents(common_ctx, frame_json),
            )
            if not self.generate_slides:
                meetings = meetings.model_copy(update={
                    "adhoc_meetings": [
                        meeting.model_copy(update={"should_generate_slides": False})
                        for meeting in meetings.adhoc_meetings
                    ],
                })

            slides = await self._generate_week_slides(common_ctx, frame_json, meetings)

            assembled = self._assemble_week_plan(
                week_num=week_num,
                week_start=week_start,
                week_end=week_end,
                frame=frame,
                meetings=meetings,
                emails=emails,
                documents=documents,
                slides=slides,
            )

            sanitized_plan = sanitize_week_plan(
                assembled.model_dump(mode="json"), registry, self.id_drift,
            )
            plan = WeekPlan.model_validate(sanitized_plan).model_dump(mode="json")
            week_plans.append(plan)

            for ev in self._plan_state_changes_to_events(plan, week_num, week_end):
                self.planning_log.append(ev)

            prev_forward_brief = plan.get("forward_brief") or plan.get(
                "week_summary", "No prior context available."
            )

        logger.info("Generated %d week plans", len(week_plans))
        return week_plans

    # ------------------------------------------------------------------
    # Per-week frame + family fanout
    # ------------------------------------------------------------------

    def _build_week_common_context(
        self,
        *,
        week_num: int,
        num_weeks: int,
        week_start: date,
        week_end: date,
        company_summary: str,
        narrative_summary: str,
        major_events: list[dict],
        current_state: WorkspaceState,
        prev_forward_brief: str,
    ) -> dict[str, str]:
        events_this_week = [
            e for e in major_events
            if _event_in_range(e, week_start, week_end)
        ]
        events_text = (
            json.dumps(events_this_week, indent=2, default=str)
            if events_this_week else "No major events this week."
        )
        return {
            "COMPANY_SUMMARY": company_summary,
            "NARRATIVE_SUMMARY": narrative_summary,
            "WEEK_NUM": str(week_num),
            "TOTAL_WEEKS": str(num_weeks),
            "WEEK_START": week_start.isoformat(),
            "WEEK_END": week_end.isoformat(),
            "STATE_SUMMARY": self.ctx.state_summary(current_state),
            "EVENTS_THIS_WEEK": events_text,
            "RECURRING_MEETINGS_SUMMARY": self._recurring_meetings_for_week(week_start),
            "PREV_FORWARD_BRIEF": prev_forward_brief,
            "WORKING_DAYS_LIST": self._working_days_for_week(week_start, week_end),
            "PEOPLE_ROSTER": self._people_roster(),
            "PROJECTS_ROSTER": self.ctx.projects_summary(current_state.projects),
        }

    async def _generate_week_frame(self, common_ctx: dict[str, str]) -> WeekFrame:
        prompt = render_prompt(self.prompts["week_frame"], common_ctx)
        raw = await self._call_llm_json(
            prompt,
            max_tokens=self.budget.week_frame,
            trace_label=f"week_{common_ctx['WEEK_NUM']}_frame",
        )
        return WeekFrame.model_validate(raw)

    async def _generate_week_meetings(
        self, common_ctx: dict[str, str], frame_json: str,
    ) -> WeekMeetingsOutput:
        variables = {**common_ctx, "WEEK_FRAME_JSON": frame_json}
        prompt = render_prompt(self.prompts["week_meetings"], variables)
        raw = await self._call_llm_json(
            prompt,
            max_tokens=self.budget.week_meetings,
            trace_label=f"week_{common_ctx['WEEK_NUM']}_meetings",
        )
        return WeekMeetingsOutput.model_validate(raw)

    async def _generate_week_emails(
        self, common_ctx: dict[str, str], frame_json: str,
    ) -> WeekEmailsOutput:
        variables = {**common_ctx, "WEEK_FRAME_JSON": frame_json}
        prompt = render_prompt(self.prompts["week_emails"], variables)
        raw = await self._call_llm_json(
            prompt,
            max_tokens=self.budget.week_emails,
            trace_label=f"week_{common_ctx['WEEK_NUM']}_emails",
        )
        return WeekEmailsOutput.model_validate(raw)

    async def _generate_week_documents(
        self, common_ctx: dict[str, str], frame_json: str,
    ) -> WeekDocumentsOutput:
        variables = {**common_ctx, "WEEK_FRAME_JSON": frame_json}
        prompt = render_prompt(self.prompts["week_documents"], variables)
        raw = await self._call_llm_json(
            prompt,
            max_tokens=self.budget.week_documents,
            trace_label=f"week_{common_ctx['WEEK_NUM']}_documents",
        )
        return WeekDocumentsOutput.model_validate(raw)

    async def _generate_week_slides(
        self,
        common_ctx: dict[str, str],
        frame_json: str,
        meetings: WeekMeetingsOutput,
    ) -> WeekSlidesOutput:
        if not self.generate_slides:
            return WeekSlidesOutput()

        target = 0
        # Cheap heuristic: if neither the frame nor any planned meeting wants
        # slides, skip the call entirely. Saves a round trip on most weeks.
        try:
            target = int(json.loads(frame_json).get("content_plan", {}).get("slides_target", 0))
        except (ValueError, KeyError, TypeError):
            target = 0
        any_slides_flag = any(m.should_generate_slides for m in meetings.adhoc_meetings)
        if target <= 0 and not any_slides_flag:
            return WeekSlidesOutput()

        meetings_json = json.dumps(meetings.model_dump(mode="json"), indent=2)
        variables = {
            **common_ctx,
            "WEEK_FRAME_JSON": frame_json,
            "WEEK_MEETINGS_JSON": meetings_json,
        }
        prompt = render_prompt(self.prompts["week_slides"], variables)
        raw = await self._call_llm_json(
            prompt,
            max_tokens=self.budget.week_slides,
            trace_label=f"week_{common_ctx['WEEK_NUM']}_slides",
        )
        return WeekSlidesOutput.model_validate(raw)

    @staticmethod
    def _assemble_week_plan(
        *,
        week_num: int,
        week_start: date,
        week_end: date,
        frame: WeekFrame,
        meetings: WeekMeetingsOutput,
        emails: WeekEmailsOutput,
        documents: WeekDocumentsOutput,
        slides: WeekSlidesOutput,
    ) -> WeekPlan:
        return WeekPlan(
            week_num=week_num,
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            week_summary=frame.week_summary,
            forward_brief=frame.forward_brief,
            adhoc_meetings=meetings.adhoc_meetings,
            meeting_outcomes=meetings.meeting_outcomes,
            email_threads=emails.email_threads,
            documents=documents.documents,
            slide_presentations=slides.slide_presentations,
            action_items_resolved=frame.action_items_resolved,
            state_changes=frame.state_changes,
        )

    def _plan_state_changes_to_events(
        self, plan: dict, week_num: int, week_end: date,
    ) -> list[WorkspaceEvent]:
        """Translate a week plan's declared state_changes into WorkspaceEvents.

        Planning intent feeds the same reducer as step 5's observed events,
        but lives in a separate event log so forecasts never contaminate
        ground truth.
        """
        ts = datetime.combine(week_end, time(17, 0))
        events: list[WorkspaceEvent] = []
        valid_statuses = {s.value for s in ProjectStatus}
        state_changes = plan.get("state_changes") or {}
        for upd in state_changes.get("project_updates") or []:
            pid = upd.get("project_id")
            new_status = upd.get("new_status")
            if pid and new_status:
                if new_status not in valid_statuses:
                    logger.warning(
                        "Week %d: dropping project_update for %r — unknown new_status %r (valid: %s)",
                        week_num, pid, new_status, sorted(valid_statuses),
                    )
                else:
                    data: dict[str, Any] = {"project_id": pid, "new_status": new_status}
                    if new_status != ProjectStatus.BLOCKED.value:
                        data["blockers"] = []
                    events.append(WorkspaceEvent(
                        type=EventType.PROJECT_STATUS_CHANGED, timestamp=ts, data=data,
                    ))
            details = upd.get("details")
            if details:
                events.append(WorkspaceEvent(
                    type=EventType.DECISION_MADE, timestamp=ts,
                    data={"description": f"[wk{week_num}] {details}"},
                ))
        return events

    def _people_roster(self) -> str:
        """Build a roster of all people with their IDs."""
        lines: list[str] = []
        for pid, person in self.org.people.items():
            lines.append(
                f"- {pid}: {person.name}, {person.role.title} "
                f"({person.role.department}, work_arrangement={person.work_arrangement.value})"
            )
        for pid, person in self.org.external_people.items():
            lines.append(f"- {pid}: {person.name}, {person.role_title} (external, {person.organization})")
        return "\n".join(lines) if lines else "No people defined."

    def _working_days_for_week(self, week_start: date, week_end: date) -> str:
        """Build a human-readable list of working days in this week."""
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        working_days = self.workspace_config.working_days
        lines: list[str] = []
        current = week_start
        while current <= week_end:
            if current.weekday() in working_days:
                lines.append(
                    f"- {current.isoformat()} ({day_names[current.weekday()]}, day_of_week={current.weekday()})"
                )
            current += timedelta(days=1)
        return "\n".join(lines) if lines else "No working days this week."

    def _recurring_meetings_for_week(self, week_start: date) -> str:
        lines: list[str] = []
        for rm in self.recurring_meetings:
            # Check if meeting occurs on any day this week
            for day_offset in range(min(5, 7)):
                d = week_start + timedelta(days=day_offset)
                if d.weekday() > 4:
                    continue
                occurs = False
                if rm.cadence == "daily":
                    occurs = True
                elif rm.cadence == "weekly" and rm.day_of_week == d.weekday():
                    occurs = True
                elif rm.cadence == "biweekly" and rm.day_of_week == d.weekday() and d.isocalendar()[1] % 2 == 0:
                    occurs = True
                elif rm.cadence == "monthly" and d.day <= 7 and (rm.day_of_week is None or rm.day_of_week == d.weekday()):
                    occurs = True
                if occurs:
                    organizer = self.org.get_person(rm.organizer_id)
                    org_name = organizer.name if organizer else rm.organizer_id
                    attendees = [
                        self.org.get_person(aid).name if self.org.get_person(aid) else aid
                        for aid in rm.attendee_ids
                    ]
                    lines.append(
                        f"- {rm.title} ({rm.meeting_type}, {rm.cadence}) on {d.strftime('%A')}: "
                        f"organizer={org_name}, attendees=[{', '.join(attendees)}], "
                        f"{rm.duration_minutes}min at {rm.time_of_day}"
                    )
                    break  # Only list each meeting once per week
        return "\n".join(lines) if lines else "No recurring meetings this week."

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_all(self, narrative: dict, week_plans: list[dict], project_trajectories: list) -> None:
        fileio.makedirs(self.output_dir)
        fileio.write_json(self.output_dir / "narrative.json", narrative)
        fileio.write_json(self.output_dir / "project_trajectories.json", project_trajectories)

        week_dir = self.output_dir / "week_plans"
        fileio.makedirs(week_dir)
        for plan in week_plans:
            fileio.write_json(week_dir / f"week_{plan['week_num']:02d}.json", plan)

        self.id_drift.persist(self.output_dir / "id_drift.json")
        fileio.write_json(
            self.output_dir / "planning_events.json",
            [ev.model_dump(mode="json") for ev in self.planning_log.events],
        )

        logger.info("Step 2 outputs saved to %s", self.output_dir)

    def _load_from_disk(self) -> dict:
        narrative = Narrative.model_validate(
            fileio.read_json(self.output_dir / "narrative.json")
        ).model_dump(mode="json")

        week_plans = []
        week_dir = self.output_dir / "week_plans"
        if week_dir.exists():
            for fname in sorted(week_dir.iterdir()):
                if fname.suffix == ".json":
                    week_plans.append(
                        WeekPlan.model_validate(fileio.read_json(fname)).model_dump(mode="json")
                    )

        traj_path = self.output_dir / "project_trajectories.json"
        project_trajectories = []
        if traj_path.exists():
            project_trajectories = validate_project_arcs(fileio.read_json(traj_path))

        logger.info("Loaded step 2: narrative + %d week plans", len(week_plans))
        return {
            "narrative": narrative,
            "week_plans": week_plans,
            "project_trajectories": project_trajectories,
        }


def _event_in_range(event: dict, start: date, end: date) -> bool:
    d = event.get("approximate_date", "")
    if not d:
        return False
    try:
        event_date = date.fromisoformat(str(d))
        return start <= event_date <= end
    except ValueError:
        return False
