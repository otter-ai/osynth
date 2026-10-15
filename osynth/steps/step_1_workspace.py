"""Step 1: Generate workspace foundation — company, org, projects, recurring meetings.

Uses the sketch-then-expand pattern:
  1. Company generation (single call)
  2. Org sketch (single call) → person detail (parallel calls)
  3. Projects sketch (single call) → project detail (parallel calls)
  4. Recurring meetings (single call)
  5. Voice assignment
  6. Initial workspace state construction
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..models.calendar import MeetingLabels, RecurringMeeting
from ..models.people import (
    DisfluencyProfile,
    ExternalPerson,
    OrgStructure,
    Person,
    Role,
    SpeakingStyle,
    Team,
)
from ..models.state import Project, ProjectStatus, SpecializedTerm, WorkspaceState
from ..models.settings import SyntheticDataSettings
from ..models.workspace import Company, WorkspaceConfig
from ..utils import fileio
from ..utils.id_validation import IdDrift, IdRegistry
from ..utils.llm_json import extract_json
from ..utils.llm_client import LLMClient, LLMResponse
from ..utils.prompt import read_prompt, render_prompt
from ..utils.prompt_trace import PromptTraceRecorder
from ..utils.voice_pool import CatalogVoiceSource, StaticVoiceSource, VoicePool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Company-size parameters
# ---------------------------------------------------------------------------

_PEOPLE_RANGE: dict[str, tuple[int, int]] = {
    "startup": (5, 8),
    "small_business": (8, 15),
    "midmarket": (10, 20),
    "enterprise": (15, -1),  # -1 → use max_speakers
}

_PROJECT_RANGE: dict[str, tuple[int, int]] = {
    "startup": (2, 3),
    "small_business": (2, 4),
    "midmarket": (3, 5),
    "enterprise": (4, 6),
}

# ---------------------------------------------------------------------------
# Sampling pools
# ---------------------------------------------------------------------------

_INDUSTRY_POOL = [
    "B2B SaaS", "Fintech", "Healthtech", "E-commerce", "Developer tools",
    "Edtech", "Cybersecurity", "Martech", "HR tech", "Logistics & supply chain",
    "Clean energy", "Legal tech", "Insurtech", "Proptech", "Biotech",
]

_DOMAIN_POOL = [
    "cloud_infrastructure", "payments", "telehealth", "marketplace", "devops",
    "online_learning", "identity_security", "analytics", "recruiting",
    "fleet_management", "solar", "contract_management", "claims_automation",
    "property_management", "genomics",
]

_SIZE_POOL = ["startup", "small_business", "midmarket", "enterprise"]

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_org_summary(org: OrgStructure) -> str:
    lines: list[str] = []
    for team in org.teams.values():
        member_names = [
            org.people[mid].name for mid in team.member_ids if mid in org.people
        ]
        manager_name = (
            org.people[team.manager_id].name if team.manager_id in org.people else "?"
        )
        lines.append(
            f"- {team.name} ({team.department}): "
            f"manager={manager_name}, members=[{', '.join(member_names)}]"
        )
    lines.append("")
    lines.append("People:")
    for person in org.people.values():
        lines.append(
            f"- {person.id}: {person.name}, {person.role.title} ({person.role.department}), "
            f"team={person.team_id}, gender={person.gender}"
        )
    if org.external_people:
        lines.append("")
        lines.append("External People:")
        for person in org.external_people.values():
            lines.append(
                f"- {person.id}: {person.name}, {person.role_title}, "
                f"organization={person.organization}, relationship={person.relationship_type}"
            )
    return "\n".join(lines)


def _build_projects_summary(projects: dict[str, Project]) -> str:
    lines: list[str] = []
    for proj in projects.values():
        lines.append(
            f"- {proj.id}: {proj.name} (status={proj.status.value}, owner={proj.owner_id}, "
            f"teams={proj.team_ids})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step implementation
# ---------------------------------------------------------------------------


class Step1Workspace:
    """Generate workspace foundation using sketch-then-expand pattern."""

    def __init__(
        self,
        settings: SyntheticDataSettings,
        output_root: str,
        *,
        llm_client: Any | None = None,
    ) -> None:
        self.settings = settings

        # LLM client (auto-detects Anthropic vs OpenAI from model name)
        self.model: str = settings.model
        self.llm = llm_client or LLMClient(model=self.model)

        # Workspace params
        self.company_size_setting: str = settings.workspace.company_size
        self.industry_setting: str = settings.workspace.industry
        self.domain_setting: str = settings.workspace.domain
        self.max_speakers: int = settings.workspace.max_speakers
        self.time_period_start = _parse_date(settings.workspace.time_period_start)
        self.time_period_end = _parse_date(settings.workspace.time_period_end)
        self.working_hours: tuple[int, int] = tuple(settings.workspace.working_hours)  # type: ignore[arg-type]
        self.working_days: list[int] = list(settings.workspace.working_days)
        self.timezone: str = settings.workspace.timezone
        self.max_concurrent: int = settings.generation.max_concurrent

        # Per-call output-token budgets. The monolithic sketches are the most
        # likely to truncate at high speaker / project counts, so configs give
        # them larger budgets than the bounded per-item expansion calls.
        self.budget = settings.generation.llm.step_1

        # Seed
        seed_val = settings.seed
        self.seed: int | None = int(seed_val) if seed_val is not None else None
        self.rng = random.Random(self.seed)

        # Output
        self.output_dir = Path(output_root) / "step_1_workspace"
        self.prompt_tracer = PromptTraceRecorder(self.output_dir / "prompt_traces")
        self._io_lock = asyncio.Lock()

        # Language
        self.language: str = settings.language

        # Prompt paths
        self._code_root = Path(__file__).resolve().parent.parent
        prompts_cfg = settings.prompts.workspace_sampling
        self.prompts = {
            "system": read_prompt(self._code_root / prompts_cfg.system, self.language),
            "company": read_prompt(self._code_root / prompts_cfg.company, self.language),
            "org_spine": read_prompt(self._code_root / prompts_cfg.org_spine, self.language),
            "team_people_sketch": read_prompt(self._code_root / prompts_cfg.team_people_sketch, self.language),
            "person_detail": read_prompt(self._code_root / prompts_cfg.person_detail, self.language),
            "external_person_detail": read_prompt(self._code_root / prompts_cfg.external_person_detail, self.language),
            "project_index": read_prompt(self._code_root / prompts_cfg.project_index, self.language),
            "project_cluster_sketch": read_prompt(self._code_root / prompts_cfg.project_cluster_sketch, self.language),
            "project_detail": read_prompt(self._code_root / prompts_cfg.project_detail, self.language),
            "recurring_meetings_company": read_prompt(self._code_root / prompts_cfg.recurring_meetings_company, self.language),
            "recurring_meetings_team": read_prompt(self._code_root / prompts_cfg.recurring_meetings_team, self.language),
            "recurring_meetings_cross": read_prompt(self._code_root / prompts_cfg.recurring_meetings_cross, self.language),
        }

        # TTS config
        self.voice_source_type: str = settings.tts.voice_source
        self.voice_cache_dir: str = settings.tts.voice_cache_dir

        # Drift tracker for LLM-emitted IDs (populated as we generate)
        self.id_drift = IdDrift()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run(self, force: bool = False) -> dict:
        # Check if already generated
        config_path = self.output_dir / "workspace_config.json"
        if config_path.exists() and not force:
            logger.info("Step 1 outputs already exist, loading from disk. Use --force to regenerate.")
            return self._load_from_disk()

        # 1. Company
        size = self._resolve("company_size", self.company_size_setting, _SIZE_POOL)
        industry = self._resolve("industry", self.industry_setting, _INDUSTRY_POOL)
        domain = self._resolve("domain", self.domain_setting, _DOMAIN_POOL)
        company = await self._generate_company(size, industry, domain)

        # 2. Org spine → parallel team people sketch → parallel person detail
        num_people = self._pick_num("people", size)
        spine = await self._generate_org_spine(company, num_people)
        org = await self._expand_org_from_spine(company, spine)

        # 3. Project index → parallel cluster sketch → parallel project detail
        num_projects = self._pick_num("projects", size)
        projects = await self._generate_projects(company, org, num_projects)

        # 4. Recurring meetings (company-wide + per-team + cross-functional in parallel)
        recurring_meetings = await self._generate_recurring_meetings(company, org, projects)

        # 5. Voice assignment
        voice_assignments = self._assign_voices(org)

        # 6. Build initial workspace state
        workspace_config = WorkspaceConfig(
            company=company,
            time_period_start=self.time_period_start,
            time_period_end=self.time_period_end,
            max_speakers=self.max_speakers,
            seed=self.seed,
            working_hours=self.working_hours,
            working_days=self.working_days,
            timezone=self.timezone,
        )

        initial_state = WorkspaceState(
            timestamp=datetime.combine(self.time_period_start, datetime.min.time()),
            projects={pid: proj for pid, proj in projects.items()},
        )

        # 7. Persist
        self._save_all(
            workspace_config=workspace_config,
            org=org,
            projects=projects,
            recurring_meetings=recurring_meetings,
            voice_assignments=voice_assignments,
            initial_state=initial_state,
        )

        return {
            "workspace_config": workspace_config,
            "org": org,
            "projects": projects,
            "recurring_meetings": recurring_meetings,
            "voice_assignments": voice_assignments,
            "initial_state": initial_state,
        }

    # ------------------------------------------------------------------
    # LLM API
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        user_prompt: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        trace_label: str = "llm_call",
    ) -> LLMResponse:
        from ..utils.usage_tracker import tracker

        response = await self.llm.create_message(
            system=self.prompts["system"],
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        tracker.track_llm_response(response)

        async with self._io_lock:
            self.prompt_tracer.record(
                label=trace_label,
                system_prompt=self.prompts["system"],
                user_prompt=user_prompt,
                response_text=response.text,
                metadata={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop_reason": response.stop_reason,
                    "output_tokens": response.output_tokens,
                },
            )
        return response

    async def _call_llm_json(
        self,
        user_prompt: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        retries: int = MAX_RETRIES,
        trace_label: str = "llm_json",
    ) -> Any:
        last_error: Exception | None = None
        last_response: LLMResponse | None = None
        for attempt in range(1 + retries):
            if attempt > 0:
                logger.warning("Retry %d/%d after JSON parse failure", attempt, retries)
            response = await self._call_llm(
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                trace_label=f"{trace_label}_attempt_{attempt + 1}",
            )
            last_response = response
            try:
                return extract_json(response.text)
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
            f"Failed to parse JSON from LLM after {1 + retries} attempts "
            f"for {trace_label}{truncation_note}"
        ) from last_error

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _resolve(self, name: str, setting: str, pool: list[str]) -> str:
        if setting == "sample":
            chosen = self.rng.choice(pool)
            logger.info("Sampled %s: %s", name, chosen)
            return chosen
        return setting

    def _pick_num(self, kind: str, size: str) -> int:
        ranges = _PEOPLE_RANGE if kind == "people" else _PROJECT_RANGE
        lo, hi = ranges[size]
        if hi == -1:
            hi = self.max_speakers
        hi = min(hi, self.max_speakers)
        lo = min(lo, hi)
        n = self.rng.randint(lo, hi)
        logger.info("Picked %d %s for %s company", n, kind, size)
        return n

    # ------------------------------------------------------------------
    # Generation: Company
    # ------------------------------------------------------------------

    async def _generate_company(self, size: str, industry: str, domain: str) -> Company:
        user_prompt = (
            self.prompts["company"]
            .replace("{{COMPANY_SIZE}}", size)
            .replace("{{COMPANY_SIZE_ENUM}}", size)
            .replace("{{INDUSTRY}}", industry)
            .replace("{{DOMAIN}}", domain)
        )
        logger.info("Generating company (size=%s, industry=%s, domain=%s)", size, industry, domain)
        data = await self._call_llm_json(
            user_prompt, max_tokens=self.budget.company, trace_label="company",
        )
        company = Company(**data)
        logger.info("Generated company: %s (%s)", company.name, company.industry)
        return company

    # ------------------------------------------------------------------
    # Generation: Org (spine → per-team people fanout → parallel person detail)
    # ------------------------------------------------------------------

    async def _generate_org_spine(self, company: Company, num_people: int) -> dict:
        company_json = company.model_dump_json(indent=2)

        org_constraints = {
            "startup": (
                "For a startup this size, keep the hierarchy flat. "
                "1-2 teams maximum. Everyone likely reports to the CEO or CTO."
            ),
            "small_business": (
                "For a small business, use 2-3 teams with 1-2 levels of hierarchy."
            ),
            "midmarket": (
                "For a mid-market company, use 3-5 teams with 2-3 levels of hierarchy."
            ),
            "enterprise": (
                "For an enterprise company, use 4-6 teams with 3-4 levels of hierarchy."
            ),
        }.get(company.size.value, "")

        user_prompt = (
            self.prompts["org_spine"]
            .replace("{{COMPANY_JSON}}", company_json)
            .replace("{{NUM_PEOPLE}}", str(num_people))
            .replace("{{ORG_CONSTRAINTS}}", org_constraints)
        )

        logger.info("Generating org spine targeting %d people", num_people)
        return await self._call_llm_json(
            user_prompt, max_tokens=self.budget.org_spine, trace_label="org_spine",
        )

    def _spine_summary(self, spine: dict) -> str:
        lines = ["Teams:"]
        for t in spine.get("teams", []):
            lines.append(
                f"- {t['id']}: {t.get('name', '')} ({t.get('department', '')}), "
                f"manager={t.get('manager_id', '')}, "
                f"target_headcount={t.get('target_headcount', '?')}, "
                f"role_mix={t.get('role_mix', '')}"
            )
        lines.append("\nManagers (already sketched):")
        for m in spine.get("managers", []):
            nickname_bits = []
            if m.get("nickname"):
                nickname_bits.append(f"nickname={m.get('nickname', '')}")
            nickname_bits.append(f"nickname_usage={m.get('nickname_usage', 'none')}")
            lines.append(
                f"- {m['id']}: {m.get('name', '')}, {m.get('role_title', '')} "
                f"({m.get('department', '')}), team={m.get('team_id', '')}, level={m.get('level', '')}, "
                f"email={m.get('email', '')}, {', '.join(nickname_bits)}"
            )
        if spine.get("external_people_sketches"):
            lines.append("\nExternal people:")
            for p in spine["external_people_sketches"]:
                nickname_bits = []
                if p.get("nickname"):
                    nickname_bits.append(f"nickname={p.get('nickname', '')}")
                nickname_bits.append(f"nickname_usage={p.get('nickname_usage', 'none')}")
                lines.append(
                    f"- {p['id']}: {p.get('name', '')}, {p.get('role_title', '')} "
                    f"at {p.get('organization', '')}, {', '.join(nickname_bits)}"
                )
        return "\n".join(lines)

    async def _generate_team_people_sketches(
        self,
        team: dict,
        spine: dict,
        spine_summary: str,
        id_start: int,
        company_summary: str,
    ) -> list[dict]:
        """Fan out: generate IC sketches for one team."""
        manager_id = team.get("manager_id", "")
        target = int(team.get("target_headcount", 0) or 0)
        ic_count = max(target - 1, 0)
        if ic_count == 0:
            return []

        id_end = id_start + ic_count - 1
        id_range = f"p{id_start}..p{id_end}"
        id_range_first = f"p{id_start}"
        reserved = sorted(
            {m["id"] for m in spine.get("managers", [])}
            | {p["id"] for p in spine.get("external_people_sketches", [])}
        )

        user_prompt = (
            self.prompts["team_people_sketch"]
            .replace("{{COMPANY_SUMMARY}}", company_summary)
            .replace("{{ORG_SPINE_SUMMARY}}", spine_summary)
            .replace("{{TEAM_JSON}}", json.dumps(team, indent=2))
            .replace("{{TEAM_ID}}", team.get("id", ""))
            .replace("{{RESERVED_IDS}}", ", ".join(reserved) or "(none)")
            .replace("{{ID_RANGE}}", id_range)
            .replace("{{ID_RANGE_FIRST}}", id_range_first)
        )
        data = await self._call_llm_json(
            user_prompt,
            max_tokens=self.budget.team_people_sketch,
            trace_label=f"team_people_sketch_{team.get('id', 'team')}",
        )
        sketches: list[dict] = list(data.get("people_sketches", []) or [])

        # Force IDs and team_id into the reserved range so siblings can't collide.
        normalized: list[dict] = []
        for offset, sketch in enumerate(sketches[:ic_count]):
            sketch = dict(sketch)
            sketch["id"] = f"p{id_start + offset}"
            sketch["team_id"] = team.get("id", "")
            if manager_id:
                sketch.setdefault("manager_id", manager_id)
            normalized.append(sketch)

        return normalized

    async def _expand_org_from_spine(self, company: Company, spine: dict) -> OrgStructure:
        """Expand spine + per-team people sketches into a full OrgStructure."""
        company_summary = (
            f"{company.name} - {company.description}\n"
            f"Industry: {company.industry} | Size: {company.size.value} | Location: {company.location}"
        )
        spine_summary = self._spine_summary(spine)

        # Pre-allocate ID slots for ICs so parallel calls can't collide.
        teams_in: list[dict] = list(spine.get("teams", []) or [])
        manager_ids = {m["id"] for m in spine.get("managers", []) or []}
        max_manager_num = 0
        for mid in manager_ids:
            m = re.match(r"p(\d+)$", mid)
            if m:
                max_manager_num = max(max_manager_num, int(m.group(1)))
        next_id = max_manager_num + 1

        team_slots: list[tuple[dict, int]] = []
        for team in teams_in:
            team_slots.append((team, next_id))
            ic_count = max(int(team.get("target_headcount", 0) or 0) - 1, 0)
            next_id += ic_count

        logger.info("Fanning out %d team people sketch calls", len(team_slots))
        per_team_sketches = await asyncio.gather(
            *[
                self._generate_team_people_sketches(
                    team, spine, spine_summary, slot, company_summary,
                )
                for team, slot in team_slots
            ]
        )

        # Combined people sketch list: managers (already detailed in spine) + ICs.
        people_sketches: list[dict] = list(spine.get("managers", []) or [])
        team_member_ids: dict[str, list[str]] = {}
        for (team, _slot), ic_sketches in zip(team_slots, per_team_sketches):
            tid = team.get("id", "")
            members = [team.get("manager_id", "")] + [s["id"] for s in ic_sketches]
            team_member_ids[tid] = [m for m in members if m]
            people_sketches.extend(ic_sketches)

        external_sketches: list[dict] = list(spine.get("external_people_sketches", []) or [])

        # Expand each sketch via existing person_detail / external_person_detail.
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def expand_person(person_sketch: dict) -> Person:
            async with semaphore:
                user_prompt = (
                    self.prompts["person_detail"]
                    .replace("{{COMPANY_SUMMARY}}", company_summary)
                    .replace("{{ORG_SKETCH_SUMMARY}}", spine_summary)
                    .replace("{{PERSON_SKETCH_JSON}}", json.dumps(person_sketch, indent=2))
                    .replace("{{PERSON_ID}}", person_sketch["id"])
                )
                data = await self._call_llm_json(
                    user_prompt,
                    max_tokens=self.budget.person_detail,
                    trace_label=f"person_detail_{person_sketch['id']}",
                )
                return _parse_person(data)

        async def expand_external_person(person_sketch: dict) -> ExternalPerson:
            async with semaphore:
                user_prompt = (
                    self.prompts["external_person_detail"]
                    .replace("{{COMPANY_SUMMARY}}", company_summary)
                    .replace("{{PERSON_SKETCH_JSON}}", json.dumps(person_sketch, indent=2))
                    .replace("{{PERSON_ID}}", person_sketch["id"])
                    .replace("{{RELATIONSHIP_TYPE}}", person_sketch.get("relationship_type", "external"))
                )
                data = await self._call_llm_json(
                    user_prompt,
                    max_tokens=self.budget.external_person_detail,
                    trace_label=f"ext_person_detail_{person_sketch['id']}",
                )
                return _parse_external_person(data)

        logger.info(
            "Expanding %d person sketches + %d external person sketches in parallel",
            len(people_sketches), len(external_sketches),
        )

        person_tasks = [expand_person(ps) for ps in people_sketches]
        ext_tasks = [expand_external_person(ps) for ps in external_sketches]

        all_results = await asyncio.gather(*person_tasks, *ext_tasks, return_exceptions=True)

        num_internal = len(people_sketches)
        people: dict[str, Person] = {}
        external_people: dict[str, ExternalPerson] = {}

        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                sketch_id = (
                    people_sketches[i]["id"]
                    if i < num_internal
                    else external_sketches[i - num_internal]["id"]
                )
                logger.error("Failed to expand person %s: %s", sketch_id, result)
                raise result
            if i < num_internal:
                assert isinstance(result, Person)
                people[result.id] = result
            else:
                assert isinstance(result, ExternalPerson)
                external_people[result.id] = result

        # Build teams with member_ids derived from the fanout results.
        teams: dict[str, Team] = {}
        for team_dict in teams_in:
            tid = team_dict.get("id", "")
            teams[tid] = Team(
                id=tid,
                name=team_dict.get("name", ""),
                department=team_dict.get("department", ""),
                purpose=team_dict.get("purpose", ""),
                manager_id=team_dict.get("manager_id", ""),
                parent_team_id=team_dict.get("parent_team_id"),
                member_ids=team_member_ids.get(tid, []),
            )

        # Reporting: spine.reporting (manager → manager) + IC → manager fallbacks.
        reporting: dict[str, str] = {
            k: v for k, v in (spine.get("reporting") or {}).items()
            if isinstance(v, str)
        }
        for team_dict in teams_in:
            tid = team_dict.get("id", "")
            mgr = team_dict.get("manager_id", "")
            if not mgr:
                continue
            for member_id in team_member_ids.get(tid, []):
                if member_id == mgr:
                    continue
                reporting.setdefault(member_id, mgr)

        org = OrgStructure(
            teams=teams,
            people=people,
            external_people=external_people,
            reporting=reporting,
        )

        logger.info(
            "Expanded org: %d people, %d external, %d teams",
            len(org.people), len(org.external_people), len(org.teams),
        )

        warnings = self._validate_org(org)
        for w in warnings:
            logger.warning("Org validation: %s", w)

        return org

    # ------------------------------------------------------------------
    # Generation: Projects (sketch → expand)
    # ------------------------------------------------------------------

    async def _generate_projects(
        self, company: Company, org: OrgStructure, num_projects: int
    ) -> dict[str, Project]:
        company_json = company.model_dump_json(indent=2)
        org_summary = _build_org_summary(org)

        # 1. Project index (clusters)
        index_prompt = (
            self.prompts["project_index"]
            .replace("{{COMPANY_JSON}}", company_json)
            .replace("{{ORG_SUMMARY}}", org_summary)
            .replace("{{NUM_PROJECTS}}", str(num_projects))
            .replace("{{START_DATE}}", self.time_period_start.isoformat())
            .replace("{{END_DATE}}", self.time_period_end.isoformat())
        )
        logger.info("Generating project index (%d projects)", num_projects)
        index_data = await self._call_llm_json(
            index_prompt, max_tokens=self.budget.project_index, trace_label="project_index",
        )

        # 2. Per-cluster sketch fanout
        clusters: list[dict] = list(index_data.get("clusters", []) or [])

        async def cluster_sketch(cluster: dict) -> list[dict]:
            cluster_prompt = (
                self.prompts["project_cluster_sketch"]
                .replace("{{COMPANY_JSON}}", company_json)
                .replace("{{ORG_SUMMARY}}", org_summary)
                .replace("{{START_DATE}}", self.time_period_start.isoformat())
                .replace("{{END_DATE}}", self.time_period_end.isoformat())
                .replace("{{CLUSTER_JSON}}", json.dumps(cluster, indent=2))
            )
            data = await self._call_llm_json(
                cluster_prompt,
                max_tokens=self.budget.project_cluster_sketch,
                trace_label=f"project_cluster_sketch_{cluster.get('id', 'cluster')}",
            )
            return list(data.get("project_sketches", []) or [])

        logger.info("Fanning out %d project cluster sketches", len(clusters))
        per_cluster = await asyncio.gather(*[cluster_sketch(c) for c in clusters])
        sketches: list[dict] = []
        for cluster_sketches in per_cluster:
            sketches.extend(cluster_sketches)

        # 3. Expand each project in parallel
        company_summary = f"{company.name} - {company.description}"
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def expand_project(project_sketch: dict) -> Project:
            async with semaphore:
                prompt = (
                    self.prompts["project_detail"]
                    .replace("{{COMPANY_SUMMARY}}", company_summary)
                    .replace("{{ORG_SUMMARY}}", org_summary)
                    .replace("{{START_DATE}}", self.time_period_start.isoformat())
                    .replace("{{END_DATE}}", self.time_period_end.isoformat())
                    .replace("{{PROJECT_SKETCH_JSON}}", json.dumps(project_sketch, indent=2))
                    .replace("{{PROJECT_ID}}", project_sketch["id"])
                )
                data = await self._call_llm_json(
                    prompt,
                    max_tokens=self.budget.project_detail,
                    trace_label=f"project_detail_{project_sketch['id']}",
                )
                project = _parse_project(data)
                if logger.isEnabledFor(logging.DEBUG):
                    surfaces = [t.surface for t in project.specialized_terms]
                    logger.debug(
                        "Project %s (%s) specialized_terms: %s",
                        project.id, project.name, surfaces or "(none)",
                    )
                return project

        logger.info("Expanding %d project sketches in parallel", len(sketches))

        results = await asyncio.gather(
            *[expand_project(ps) for ps in sketches], return_exceptions=True
        )

        projects: dict[str, Project] = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Failed to expand project %s: %s", sketches[i].get("id"), result)
                raise result
            projects[result.id] = result

        # Validate project owner_ids against the org
        people_only_registry = IdRegistry.from_org(org)
        for proj in projects.values():
            proj.owner_id = self.id_drift.resolve_person_id(
                proj.owner_id,
                people_only_registry,
                field_name="owner_id",
                context=f"project:{proj.id}",
            )

        logger.info("Generated %d projects", len(projects))
        return projects

    # ------------------------------------------------------------------
    # Generation: Recurring meetings
    # ------------------------------------------------------------------

    async def _generate_recurring_meetings(
        self,
        company: Company,
        org: OrgStructure,
        projects: dict[str, Project],
    ) -> list[RecurringMeeting]:
        company_json = company.model_dump_json(indent=2)
        org_summary = _build_org_summary(org)
        projects_summary = _build_projects_summary(projects)
        common_subs = {
            "{{COMPANY_JSON}}": company_json,
            "{{ORG_SUMMARY}}": org_summary,
            "{{PROJECTS_SUMMARY}}": projects_summary,
            "{{WORKING_HOURS_START}}": str(self.working_hours[0]),
            "{{WORKING_HOURS_END}}": str(self.working_hours[1]),
            "{{TIMEZONE}}": self.timezone,
        }

        def render(template: str, extras: dict[str, str] | None = None) -> str:
            return render_prompt(template, {**common_subs, **(extras or {})})

        async def call(template_key: str, max_tokens: int, label: str, extras: dict[str, str] | None = None) -> list[dict]:
            data = await self._call_llm_json(
                render(self.prompts[template_key], extras),
                max_tokens=max_tokens,
                trace_label=label,
            )
            return list(data.get("recurring_meetings", []) or [])

        async def team_call(team: Team) -> list[dict]:
            return await call(
                "recurring_meetings_team",
                self.budget.recurring_meetings_team,
                f"recurring_meetings_team_{team.id}",
                extras={
                    "{{TEAM_JSON}}": json.dumps(team.model_dump(), indent=2),
                    "{{TEAM_ID}}": team.id,
                },
            )

        logger.info(
            "Generating recurring meetings (1 company-wide + %d per-team + 1 cross-functional)",
            len(org.teams),
        )
        company_task = call(
            "recurring_meetings_company",
            self.budget.recurring_meetings_company,
            "recurring_meetings_company",
        )
        team_tasks = [team_call(team) for team in org.teams.values()]
        cross_task = call(
            "recurring_meetings_cross",
            self.budget.recurring_meetings_cross,
            "recurring_meetings_cross",
        )

        all_results = await asyncio.gather(company_task, *team_tasks, cross_task)
        raw_meetings: list[dict] = []
        for batch in all_results:
            raw_meetings.extend(batch)

        registry = IdRegistry.from_org(org, projects)
        meetings: list[RecurringMeeting] = []
        for m in raw_meetings:
            title = m.get("title", "<recurring>")
            ctx = f"recurring:{title}"
            m["attendee_ids"] = self.id_drift.filter_person_ids(
                m.get("attendee_ids", []) or [],
                registry,
                field_name="attendee_ids",
                context=ctx,
            )
            m["organizer_id"] = self.id_drift.resolve_person_id(
                m.get("organizer_id", "") or "",
                registry,
                field_name="organizer_id",
                context=ctx,
                fallback=m["attendee_ids"][0] if m["attendee_ids"] else None,
            )
            if "labels" in m and isinstance(m["labels"], dict):
                m["labels"] = MeetingLabels(**m["labels"])
            meetings.append(RecurringMeeting(**m))

        logger.info("Generated %d recurring meetings total", len(meetings))
        return meetings

    # ------------------------------------------------------------------
    # Voice assignment
    # ------------------------------------------------------------------

    def _assign_voices(self, org: OrgStructure) -> dict[str, str]:
        if self.voice_source_type in ("commonvoice", "catalog"):
            source = CatalogVoiceSource(self.voice_cache_dir)
        else:
            source = StaticVoiceSource()

        pool = VoicePool(source, seed=self.seed)
        all_people = org.all_people()

        try:
            assignments = pool.assign_all(all_people)
        except RuntimeError:
            logger.warning("No voice files found; voice assignment skipped.")
            assignments = {}

        # Apply assignments to person models
        for pid, voice_id in assignments.items():
            person = org.get_person(pid)
            if person:
                person.voice_id = voice_id

        logger.info("Assigned voices to %d people", len(assignments))
        return assignments

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_org(self, org: OrgStructure) -> list[str]:
        warnings: list[str] = []
        all_ids = set(org.people) | set(org.external_people)

        # Check team manager exists
        for team_id, team in org.teams.items():
            if team.manager_id not in org.people:
                warnings.append(f"Team {team_id} manager {team.manager_id} not in people")
            for mid in team.member_ids:
                if mid not in all_ids:
                    warnings.append(f"Team {team_id} member {mid} not found")

        # Check reporting tree
        for person_id, manager_id in org.reporting.items():
            if person_id not in all_ids:
                warnings.append(f"Reporting: {person_id} not found")
            if manager_id not in all_ids:
                warnings.append(f"Reporting: manager {manager_id} not found")

        # Check all people belong to a team
        people_in_teams = set()
        for team in org.teams.values():
            people_in_teams.update(team.member_ids)
        for pid in org.people:
            if pid not in people_in_teams:
                warnings.append(f"Person {pid} not assigned to any team")

        return warnings

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_all(
        self,
        workspace_config: WorkspaceConfig,
        org: OrgStructure,
        projects: dict[str, Project],
        recurring_meetings: list[RecurringMeeting],
        voice_assignments: dict[str, str],
        initial_state: WorkspaceState,
    ) -> None:
        fileio.makedirs(self.output_dir)
        fileio.write_json(self.output_dir / "workspace_config.json", workspace_config.model_dump(mode="json"))
        fileio.write_json(self.output_dir / "org.json", org.model_dump(mode="json"))
        fileio.write_json(
            self.output_dir / "projects.json",
            [p.model_dump(mode="json") for p in projects.values()],
        )
        fileio.write_json(
            self.output_dir / "recurring_meetings.json",
            [m.model_dump(mode="json") for m in recurring_meetings],
        )
        fileio.write_json(self.output_dir / "voice_assignments.json", voice_assignments)
        fileio.write_json(self.output_dir / "workspace_state.json", initial_state.model_dump(mode="json"))
        self.id_drift.persist(self.output_dir / "id_drift.json")

        logger.info("Step 1 outputs saved to %s", self.output_dir)

    def _load_from_disk(self) -> dict:
        """Load previously generated step 1 data."""
        with open(self.output_dir / "workspace_config.json") as f:
            workspace_config = WorkspaceConfig.model_validate(json.load(f))
        with open(self.output_dir / "org.json") as f:
            org = OrgStructure.model_validate(json.load(f))
        with open(self.output_dir / "projects.json") as f:
            projects_data = json.load(f)
            if isinstance(projects_data, dict):
                projects_data = list(projects_data.values())
            projects = {p["id"]: Project.model_validate(p) for p in projects_data}
        with open(self.output_dir / "recurring_meetings.json") as f:
            recurring_meetings = [RecurringMeeting.model_validate(rm) for rm in json.load(f)]
        with open(self.output_dir / "voice_assignments.json") as f:
            voice_assignments = json.load(f)
        with open(self.output_dir / "workspace_state.json") as f:
            initial_state = WorkspaceState.model_validate(json.load(f))

        logger.info("Loaded step 1 from disk: %d people, %d projects, %d meetings",
                     len(org.people), len(projects), len(recurring_meetings))

        return {
            "workspace_config": workspace_config,
            "org": org,
            "projects": projects,
            "recurring_meetings": recurring_meetings,
            "voice_assignments": voice_assignments,
            "initial_state": initial_state,
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_person(data: dict) -> Person:
    if isinstance(data.get("role"), dict):
        data["role"] = Role(**data["role"])
    if isinstance(data.get("speaking_style"), dict):
        ss = data["speaking_style"]
        if isinstance(ss.get("disfluency"), dict):
            ss["disfluency"] = DisfluencyProfile(**ss["disfluency"])
        data["speaking_style"] = SpeakingStyle(**ss)
    return Person(**data)


def _parse_external_person(data: dict) -> ExternalPerson:
    if isinstance(data.get("speaking_style"), dict):
        ss = data["speaking_style"]
        if isinstance(ss.get("disfluency"), dict):
            ss["disfluency"] = DisfluencyProfile(**ss["disfluency"])
        data["speaking_style"] = SpeakingStyle(**ss)
    return ExternalPerson(**data)


def _parse_project(data: dict) -> Project:
    if isinstance(data.get("start_date"), str):
        data["start_date"] = date.fromisoformat(data["start_date"])
    if isinstance(data.get("target_end_date"), str):
        data["target_end_date"] = date.fromisoformat(data["target_end_date"])
    if isinstance(data.get("status"), str):
        data["status"] = ProjectStatus(data["status"])
    raw_terms = data.get("specialized_terms") or []
    parsed_terms: list[SpecializedTerm] = []
    for entry in raw_terms:
        if isinstance(entry, str):
            parsed_terms.append(SpecializedTerm(surface=entry))
        elif isinstance(entry, dict) and entry.get("surface"):
            parsed_terms.append(SpecializedTerm(**entry))
    data["specialized_terms"] = parsed_terms
    return Project(**data)


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
