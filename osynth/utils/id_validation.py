"""Cross-step validation of LLM-emitted IDs against known org entities.

LLM JSON output frequently references organizational entities (people, projects)
by ID. Models occasionally hallucinate IDs that do not exist in the workspace,
which corrupts downstream artifact generation. This module centralizes
validation so every step that consumes such IDs filters through the same
helpers, logs drift through the same logger, and produces an inspectable
``id_drift.json`` artifact.

Typical usage in a step::

    registry = IdRegistry.from_org(self.org, self.projects)
    drift = IdDrift()
    sanitize_week_plan(plan, registry, drift)
    drift.persist(self.output_dir / "id_drift.json")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models.people import OrgStructure
from ..models.state import Project
from . import fileio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdRegistry:
    """The set of IDs that may legitimately appear in LLM output."""

    person_ids: frozenset[str]
    project_ids: frozenset[str]

    @classmethod
    def from_org(
        cls,
        org: OrgStructure,
        projects: dict[str, Project] | None = None,
    ) -> IdRegistry:
        people = set(org.people.keys()) | set(org.external_people.keys())
        proj_ids = set((projects or {}).keys())
        return cls(person_ids=frozenset(people), project_ids=frozenset(proj_ids))

    def has_person(self, pid: str) -> bool:
        return pid in self.person_ids

    def has_project(self, pid: str) -> bool:
        return pid in self.project_ids


@dataclass
class IdDrift:
    """Accumulates ID-validation events across one step's LLM outputs."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        kind: str,
        field_name: str,
        context: str,
        bad: list[str] | str,
        action: str,
    ) -> None:
        self.entries.append({
            "kind": kind,
            "field": field_name,
            "context": context,
            "bad": bad,
            "action": action,
        })
        logger.warning(
            "id_drift: %s %s in %s — %s (%s)",
            kind, field_name, context, bad, action,
        )

    def filter_person_ids(
        self,
        ids: list[str],
        registry: IdRegistry,
        *,
        field_name: str,
        context: str,
    ) -> list[str]:
        """Return only IDs known to the org; record any drops."""
        valid: list[str] = []
        invalid: list[str] = []
        for pid in ids:
            if registry.has_person(pid):
                valid.append(pid)
            else:
                invalid.append(pid)
        if invalid:
            self.record(
                kind="person",
                field_name=field_name,
                context=context,
                bad=invalid,
                action="dropped",
            )
        return valid

    def resolve_person_id(
        self,
        pid: str,
        registry: IdRegistry,
        *,
        field_name: str,
        context: str,
        fallback: str | None = None,
    ) -> str:
        """Return a known person ID or the fallback (also recorded as drift)."""
        if not pid:
            return fallback or ""
        if registry.has_person(pid):
            return pid
        action = f"replaced_with={fallback}" if fallback else "set_to_empty"
        self.record(
            kind="person",
            field_name=field_name,
            context=context,
            bad=pid,
            action=action,
        )
        return fallback or ""

    def normalize_project_id(
        self,
        pid: str | None,
        registry: IdRegistry,
        *,
        field_name: str,
        context: str,
    ) -> str | None:
        """Return the project ID if it exists, otherwise None (recorded as drift)."""
        if not pid:
            return None
        if registry.has_project(pid):
            return pid
        self.record(
            kind="project",
            field_name=field_name,
            context=context,
            bad=pid,
            action="set_to_null",
        )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_drift_events": len(self.entries),
            "by_kind": _counts_by_key(self.entries, "kind"),
            "by_field": _counts_by_key(self.entries, "field"),
            "events": self.entries,
        }

    def persist(self, path: Path) -> None:
        fileio.makedirs(path.parent)
        fileio.write_json(path, self.to_dict())


def _counts_by_key(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e[key]] = counts.get(e[key], 0) + 1
    return counts


def dedupe_preserve_order(ids: list[str]) -> list[str]:
    """Return unique IDs in first-seen order."""
    return list(dict.fromkeys(ids))


# ----------------------------------------------------------------------
# Step-2 week-plan sanitization
# ----------------------------------------------------------------------


def sanitize_week_plan(
    plan: dict[str, Any],
    registry: IdRegistry,
    drift: IdDrift,
) -> dict[str, Any]:
    """Validate person/project IDs in a week_plan dict in place.

    The week_plan schema is assembled from the step-2 frame + family fanout
    (see ``osynth/models/planning.py``). Any ID that does not exist
    in ``registry`` is filtered (lists) or nulled (scalars). Mutates ``plan``
    and returns it for chaining.
    """
    week_label = f"week_{plan.get('week_num', '?')}"

    for adhoc in plan.get("adhoc_meetings", []) or []:
        title = adhoc.get("title", "<adhoc>")
        ctx = f"{week_label}/adhoc:{title}"
        adhoc["attendee_ids"] = drift.filter_person_ids(
            adhoc.get("attendee_ids", []) or [],
            registry,
            field_name="attendee_ids",
            context=ctx,
        )
        fallback = adhoc["attendee_ids"][0] if adhoc["attendee_ids"] else None
        adhoc["organizer_id"] = drift.resolve_person_id(
            adhoc.get("organizer_id", "") or "",
            registry,
            field_name="organizer_id",
            context=ctx,
            fallback=fallback,
        )
        adhoc["related_project_id"] = drift.normalize_project_id(
            adhoc.get("related_project_id"),
            registry,
            field_name="related_project_id",
            context=ctx,
        )

    for email in plan.get("email_threads", []) or []:
        subject = email.get("subject", "<email>")
        ctx = f"{week_label}/email:{subject}"
        email["participant_ids"] = dedupe_preserve_order(drift.filter_person_ids(
            email.get("participant_ids", []) or [],
            registry,
            field_name="participant_ids",
            context=ctx,
        ))
        fallback = email["participant_ids"][0] if email["participant_ids"] else None
        email["initiator_id"] = drift.resolve_person_id(
            email.get("initiator_id", "") or "",
            registry,
            field_name="initiator_id",
            context=ctx,
            fallback=fallback,
        )
        if email["initiator_id"] and email["initiator_id"] not in email["participant_ids"]:
            email["participant_ids"] = [email["initiator_id"], *email["participant_ids"]]
            drift.record(
                kind="person",
                field_name="participant_ids",
                context=ctx,
                bad=email["initiator_id"],
                action="prepended_initiator",
            )
        email["cc_ids"] = dedupe_preserve_order(drift.filter_person_ids(
            email.get("cc_ids", []) or [],
            registry,
            field_name="cc_ids",
            context=ctx,
        ))
        overlapping_cc_ids = [pid for pid in email["cc_ids"] if pid in email["participant_ids"]]
        if overlapping_cc_ids:
            drift.record(
                kind="person",
                field_name="cc_ids",
                context=ctx,
                bad=overlapping_cc_ids,
                action="dropped_overlap_with_participant_ids",
            )
            email["cc_ids"] = [pid for pid in email["cc_ids"] if pid not in email["participant_ids"]]
        email["related_project_id"] = drift.normalize_project_id(
            email.get("related_project_id"),
            registry,
            field_name="related_project_id",
            context=ctx,
        )

    for doc in plan.get("documents", []) or []:
        title = doc.get("title", "<document>")
        ctx = f"{week_label}/doc:{title}"
        doc["author_id"] = drift.resolve_person_id(
            doc.get("author_id", "") or "",
            registry,
            field_name="author_id",
            context=ctx,
        )
        doc["related_project_id"] = drift.normalize_project_id(
            doc.get("related_project_id"),
            registry,
            field_name="related_project_id",
            context=ctx,
        )

    for slide in plan.get("slide_presentations", []) or []:
        title = slide.get("title", "<slides>")
        ctx = f"{week_label}/slides:{title}"
        slide["author_id"] = drift.resolve_person_id(
            slide.get("author_id", "") or "",
            registry,
            field_name="author_id",
            context=ctx,
        )

    state_changes = plan.get("state_changes") or {}
    for update in state_changes.get("project_updates", []) or []:
        ctx = f"{week_label}/state_changes/project_update"
        update["project_id"] = drift.normalize_project_id(
            update.get("project_id"),
            registry,
            field_name="project_id",
            context=ctx,
        )

    return plan
