"""Context builder for LLM prompts.

Different pipeline steps need different slices of context. This module
provides composable building blocks rather than a single monolithic context
string. Each method produces a self-contained text snippet; callers pick
which pieces to include.
"""
from __future__ import annotations

import unicodedata

from ..models.artifacts import MeetingArtifact
from ..models.people import ExternalPerson, NicknameUsage, OrgStructure, Person
from ..models.state import Project, WorkspaceState
from ..models.workspace import WorkspaceConfig


def _normalize_person_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name or "")
    return " ".join(normalized.split()).casefold()


def person_reference_name(person: Person | ExternalPerson) -> str:
    if person.nickname and person.nickname_usage == NicknameUsage.MOSTLY:
        return person.nickname
    return person.name


class ContextBuilder:
    """Assembles relevant context for LLM generation calls.

    Usage example (in a meeting generation step)::

        ctx = ContextBuilder(workspace, org)
        prompt_context = "\\n\\n".join([
            ctx.company_summary(),
            ctx.participants_summary(attendee_ids),
            ctx.state_summary(state),
        ])

    For the disfluency injection step, use ``person_speaking_profile``
    instead — it includes the DisfluencyProfile details that other steps
    don't need.
    """

    def __init__(self, workspace: WorkspaceConfig, org: OrgStructure):
        self.workspace = workspace
        self.org = org

    # ------------------------------------------------------------------
    # Company / org
    # ------------------------------------------------------------------

    def company_summary(self) -> str:
        c = self.workspace.company
        return (
            f"{c.name} - {c.description}\n"
            f"Industry: {c.industry} | Size: {c.size.value} | Location: {c.location}\n"
            f"Products: {', '.join(c.products)}\n"
            f"Current situation: {c.current_situation}"
        )

    def org_summary(self) -> str:
        parts = []
        for team_id, team in self.org.teams.items():
            manager = self.org.get_person(team.manager_id)
            manager_name = manager.name if manager else "Unknown"
            members = [
                self.org.people[mid].name
                for mid in team.member_ids
                if mid in self.org.people and mid != team.manager_id
            ]
            parts.append(
                f"- {team.name} ({team.department}): "
                f"Manager: {manager_name}, "
                f"Members: {', '.join(members)}"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # People — basic (for most steps)
    # ------------------------------------------------------------------

    def person_summary(self, person_id: str) -> str:
        """Name, role, team, expertise, personality. No speaking details."""
        person = self.org.get_person(person_id)
        if not person:
            return f"Unknown person: {person_id}"
        identity_lines = [f"  Person ID: {person.id}"]
        if person.nickname:
            identity_lines.append(f"  Nickname: {person.nickname}")
        identity_lines.append(f"  Nickname usage: {person.nickname_usage.value}")
        identity_lines.append(f"  Work arrangement: {person.work_arrangement.value}")
        identity_block = "\n".join(identity_lines)
        if isinstance(person, Person):
            team = self.org.teams.get(person.team_id)
            team_name = team.name if team else "Unknown team"
            return (
                f"{person.name} ({person.role.title}, {team_name})\n"
                f"{identity_block}\n"
                f"  Level: {person.role.level}\n"
                f"  Expertise: {', '.join(person.expertise)}\n"
                f"  Personality: {', '.join(person.personality_traits)}"
            )
        return (
            f"{person.name} ({person.role_title or person.relationship_type}, "
            f"external from {person.organization})\n"
            f"{identity_block}\n"
            f"  Relationship: {person.relationship_type}\n"
            f"  Expertise: {', '.join(person.expertise)}\n"
            f"  Personality: {', '.join(person.personality_traits)}"
        )

    def participants_summary(self, person_ids: list[str]) -> str:
        return "\n".join(f"- {self.person_summary(pid)}" for pid in person_ids)

    # ------------------------------------------------------------------
    # People — speaking details (for transcript generation / disfluency)
    # ------------------------------------------------------------------

    def person_speaking_profile(self, person_id: str) -> str:
        """Full speaking profile including style and disfluency parameters.

        Use this only in steps that generate or modify transcripts.
        """
        person = self.org.get_person(person_id)
        if not person:
            return f"Unknown person: {person_id}"

        lines = [self.person_summary(person_id)]
        style = person.speaking_style
        if style:
            style_parts = []
            if style.pace:
                style_parts.append(f"pace={style.pace}")
            if style.verbosity:
                style_parts.append(f"verbosity={style.verbosity}")
            if style.interruption_tendency:
                style_parts.append(f"interrupts={style.interruption_tendency}")
            if style.accent:
                style_parts.append(f"accent={style.accent}")
            if style_parts:
                lines.append(f"  Speaking style: {', '.join(style_parts)}")
            if style.notable_patterns:
                lines.append(f"  Speech patterns: {', '.join(style.notable_patterns)}")
            if style.disfluency:
                d = style.disfluency
                disf_parts = []
                if d.filler_words:
                    disf_parts.append(f"fillers={d.filler_words} @ {d.filler_rate}/100w")
                if d.repetition_rate:
                    disf_parts.append(f"repetitions={d.repetition_rate}/100w")
                if d.restart_rate:
                    disf_parts.append(f"restarts={d.restart_rate}/100w")
                if d.truncation_rate:
                    disf_parts.append(f"truncations={d.truncation_rate}/100w")
                if d.hedge_words:
                    disf_parts.append(f"hedges={d.hedge_words}")
                if d.backchannel_frequency:
                    disf_parts.append(f"backchannels={d.backchannel_frequency}")
                if d.backchannel_phrases:
                    disf_parts.append(f"backchannel_phrases={d.backchannel_phrases}")
                if disf_parts:
                    lines.append(f"  Disfluency: {', '.join(disf_parts)}")

        return "\n".join(lines)

    def participants_speaking_profiles(self, person_ids: list[str]) -> str:
        return "\n".join(f"- {self.person_speaking_profile(pid)}" for pid in person_ids)

    # ------------------------------------------------------------------
    # Projects / state
    # ------------------------------------------------------------------

    def project_summary(self, project_id: str, state: WorkspaceState) -> str:
        project = state.projects.get(project_id)
        if not project:
            return f"Unknown project: {project_id}"
        owner = self.org.get_person(project.owner_id)
        owner_name = owner.name if owner else "Unknown"
        blockers = ", ".join(project.current_blockers) if project.current_blockers else "None"
        lines = [
            f"{project.name} (Status: {project.status.value})",
            f"  Owner: {owner_name}",
            f"  Description: {project.description}",
            f"  Blockers: {blockers}",
        ]
        if project.specialized_terms:
            term_strs: list[str] = []
            for t in project.specialized_terms:
                if t.expansion:
                    term_strs.append(f"{t.surface} ({t.expansion})")
                else:
                    term_strs.append(t.surface)
            lines.append(f"  Specialized vocabulary: {', '.join(term_strs)}")
        return "\n".join(lines)

    def all_projects_summary(self, state: WorkspaceState) -> str:
        summaries = [f"- {self.project_summary(pid, state)}" for pid in state.projects]
        return "\n".join(summaries) if summaries else "No active projects."

    def projects_summary(self, projects: dict[str, Project]) -> str:
        """Compact roster of project IDs → names + owner. For prompts that
        need the LLM to emit ``related_project_id`` references."""
        if not projects:
            return "No projects defined."
        lines: list[str] = []
        for pid, project in projects.items():
            owner = self.org.get_person(project.owner_id)
            owner_name = owner.name if owner else project.owner_id or "unassigned"
            lines.append(
                f"- {pid}: {project.name} (owner: {owner_name}, status: {project.status.value})"
            )
        return "\n".join(lines)

    def state_summary(self, state: WorkspaceState) -> str:
        parts = ["Active Projects:"]
        for pid, project in state.projects.items():
            parts.append(f"  - {project.name}: {project.status.value}")
            if project.current_blockers:
                parts.append(f"    Blockers: {', '.join(project.current_blockers)}")

        open_items = [ai for ai in state.action_items.values() if ai.status == "open"]
        if open_items:
            parts.append(f"\nOpen Action Items ({len(open_items)} total):")
            for item in open_items[:10]:
                assignee = self.org.get_person(item.assignee_id)
                name = assignee.name if assignee else item.assignee_id
                parts.append(f"  - [{name}] {item.description}")

        if state.recent_decisions:
            parts.append("\nRecent Decisions:")
            for dec in state.recent_decisions[-5:]:
                parts.append(f"  - {dec}")

        return "\n".join(parts)

    def action_items_for(self, person_ids: list[str], state: WorkspaceState) -> str:
        items = []
        for ai in state.action_items.values():
            if ai.status == "open" and ai.assignee_id in person_ids:
                assignee = self.org.get_person(ai.assignee_id)
                name = assignee.name if assignee else ai.assignee_id
                items.append(f"- [{name}] {ai.description}")
        return "\n".join(items) if items else "No open action items."

    # ------------------------------------------------------------------
    # Meeting history
    # ------------------------------------------------------------------

    def meeting_history(
        self,
        person_ids: list[str],
        meetings: list[MeetingArtifact],
        limit: int = 3,
    ) -> str:
        relevant = [m for m in meetings if m.summary]
        recent = relevant[-limit:] if len(relevant) > limit else relevant
        if not recent:
            return "No prior meeting history."
        return "\n".join(f"- {m.summary}" for m in recent)

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def resolve_person_by_name(
        self, name: str, candidate_ids: list[str] | None = None
    ) -> str | None:
        normalized_target = _normalize_person_name(name)
        if not normalized_target:
            return None
        search_in = candidate_ids or list(self.org.people.keys())
        for pid in search_in:
            person = self.org.get_person(pid)
            if person and _normalize_person_name(person.name) == normalized_target:
                return pid
        return None
