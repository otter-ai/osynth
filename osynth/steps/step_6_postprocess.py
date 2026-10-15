"""Step 6: Post-processing — generate derived artifacts from reconciled state.

Produces:
  - Consolidated action item list
  - Per-meeting summaries and outlines (LLM-generated)
  - Consistency checks on artifact relationships
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ..models.llm_outputs import JargonExtractionResponse
from ..models.people import OrgStructure
from ..models.settings import SyntheticDataSettings
from ..models.state import Project, WorkspaceState
from ..models.workspace import WorkspaceConfig
from ..utils import fileio
from ..utils.context_builder import ContextBuilder
from ..utils.language import LANG_CONFIG
from ..utils.llm_client import LLMClient, model_uses_reasoning
from ..utils.llm_json import extract_json
from ..utils.prompt import read_prompt

logger = logging.getLogger(__name__)

MAX_RETRIES = 1


class Step6PostProcess:
    """Generate summaries, outlines, and consolidated action items."""

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
        self.max_concurrent: int = settings.generation.max_concurrent

        self.output_dir = Path(output_root) / "step_6_postprocess"
        self.workspace_config: WorkspaceConfig = data["workspace_config"]
        self.org: OrgStructure = data["org"]
        self.final_state: WorkspaceState = data["final_state"]
        self.artifacts: dict[str, dict] = data.get("artifacts", {})
        self.projects: dict[str, Project] = (
            data.get("projects") or self.final_state.projects or {}
        )
        self.ctx = ContextBuilder(self.workspace_config, self.org)
        self.language: str = settings.language
        lang_cfg = LANG_CONFIG.get(self.language, LANG_CONFIG["en"])
        self._lang_name: str = lang_cfg["name"]

        code_root = Path(__file__).resolve().parent.parent
        self._jargon_prompt: str = read_prompt(
            code_root / settings.prompts.postprocess.jargon_extraction,
            self.language,
        )

    async def run(self, force: bool = False) -> dict:
        items_path = self.output_dir / "action_items.json"
        if items_path.exists() and not force:
            logger.info("Step 6 outputs exist, loading from disk.")
            return self._load_from_disk()

        fileio.makedirs(self.output_dir)
        fileio.makedirs(self.output_dir / "summaries")
        fileio.makedirs(self.output_dir / "outlines")
        fileio.makedirs(self.output_dir / "jargon")

        # 1. Consolidated action items
        action_items = self._consolidate_action_items()
        fileio.write_json(self.output_dir / "action_items.json", action_items)
        logger.info("Consolidated %d action items", len(action_items))

        # 2. Generate meeting summaries and outlines
        meeting_artifacts = {
            aid: art for aid, art in self.artifacts.items()
            if art.get("type") == "meeting" and art.get("realized_status") == "completed"
        }

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_meeting(meeting_id: str, meeting: dict) -> None:
            async with semaphore:
                await self._generate_meeting_summary(meeting_id, meeting)
                await self._generate_meeting_outline(meeting_id, meeting)

        logger.info("Generating summaries/outlines for %d meetings", len(meeting_artifacts))
        await asyncio.gather(
            *[process_meeting(mid, m) for mid, m in meeting_artifacts.items()],
            return_exceptions=True,
        )

        # 3. Post-hoc jargon extraction (flat surface + concept-cluster views)
        jargon_by_artifact = await self._extract_jargon_across_artifacts(semaphore)
        self._write_workspace_jargon_aggregate(jargon_by_artifact)

        # 4. Consistency checks
        report = self._run_consistency_checks()
        fileio.write_json(self.output_dir / "consistency_report.json", report)
        self._write_consistency_markdown(report)

        logger.info(
            "Step 6 complete: %d items, %d meetings processed, %d jargon extractions, "
            "%d issues found",
            len(action_items), len(meeting_artifacts), len(jargon_by_artifact),
            len(report.get("issues", [])),
        )

        return {"action_items": action_items}

    # ------------------------------------------------------------------
    # Action items
    # ------------------------------------------------------------------

    def _consolidate_action_items(self) -> list[dict]:
        items: list[dict] = []
        for ai_id, ai in self.final_state.action_items.items():
            assignee = self.org.get_person(ai.assignee_id)
            items.append({
                "id": ai_id,
                "description": ai.description,
                "assignee_id": ai.assignee_id,
                "assignee_name": assignee.name if assignee else ai.assignee_id,
                "status": ai.status,
                "due_date": str(ai.due_date) if ai.due_date else None,
                "source_meeting_id": ai.source_meeting_id,
                "created_at": str(ai.created_at),
            })
        return items

    # ------------------------------------------------------------------
    # Meeting summaries and outlines
    # ------------------------------------------------------------------

    async def _generate_meeting_summary(self, meeting_id: str, meeting: dict) -> None:
        transcript_text = meeting.get("transcript_text", "")
        if not transcript_text:
            return

        lang_note = f"Write the summary in {self._lang_name}.\n" if self.language != "en" else ""
        prompt = (
            f"{lang_note}Write a concise 2-4 sentence summary of this meeting.\n\n"
            f"Meeting: {meeting.get('title', '')}\n"
            f"Transcript:\n{transcript_text[:8000]}\n\n"
            f"Summary:"
        )

        try:
            summary = await self._call_llm(prompt, max_tokens=500)
            path = self.output_dir / "summaries" / f"{meeting_id}_summary.md"
            fileio.write_text(path, f"# {meeting.get('title', '')}\n\n{summary}\n")
        except Exception as e:
            logger.warning("Failed to generate summary for %s: %s", meeting_id, e)

    async def _generate_meeting_outline(self, meeting_id: str, meeting: dict) -> None:
        transcript_text = meeting.get("transcript_text", "")
        if not transcript_text:
            return

        lang_note = f"Write the outline in {self._lang_name}.\n" if self.language != "en" else ""
        prompt = (
            f"{lang_note}Create a structured outline of this meeting including:\n"
            f"- Topics discussed (with approximate timestamps)\n"
            f"- Key decisions made\n"
            f"- Action items assigned\n\n"
            f"Meeting: {meeting.get('title', '')}\n"
            f"Duration: {meeting.get('start_time', '')} to {meeting.get('end_time', '')}\n"
            f"Transcript:\n{transcript_text[:8000]}\n\n"
            f"Outline:"
        )

        try:
            outline = await self._call_llm(prompt, max_tokens=4000)
            path = self.output_dir / "outlines" / f"{meeting_id}_outline.md"
            fileio.write_text(path, f"# {meeting.get('title', '')} — Outline\n\n{outline}\n")
        except Exception as e:
            logger.warning("Failed to generate outline for %s: %s", meeting_id, e)

    # ------------------------------------------------------------------
    # Jargon extraction
    # ------------------------------------------------------------------

    _JARGON_TEXT_LIMIT = 12000

    async def _extract_jargon_across_artifacts(
        self, semaphore: asyncio.Semaphore
    ) -> dict[str, dict]:
        artifacts = list(self.artifacts.items())
        if not artifacts:
            return {}

        async def extract_one(artifact_id: str, artifact: dict) -> tuple[str, dict] | None:
            text = self._artifact_text_for_jargon(artifact)
            if not text.strip():
                return None
            async with semaphore:
                try:
                    result = await self._extract_jargon_for_artifact(artifact_id, artifact, text)
                except Exception as e:
                    logger.warning(
                        "Failed jargon extraction for %s: %s", artifact_id, e,
                    )
                    return None
            return artifact_id, result

        logger.info("Extracting jargon from %d artifacts", len(artifacts))
        results = await asyncio.gather(
            *[extract_one(aid, art) for aid, art in artifacts],
            return_exceptions=False,
        )
        by_artifact: dict[str, dict] = {}
        for entry in results:
            if entry is None:
                continue
            artifact_id, payload = entry
            by_artifact[artifact_id] = payload
            fileio.write_json(
                self.output_dir / "jargon" / f"{artifact_id}.json", payload,
            )
        return by_artifact

    def _artifact_text_for_jargon(self, artifact: dict) -> str:
        kind = artifact.get("type")
        if kind == "meeting":
            return artifact.get("transcript_text", "") or ""
        if kind == "email_thread":
            chunks: list[str] = []
            for msg in artifact.get("messages", []):
                subject = msg.get("subject", "")
                body = msg.get("body", "")
                chunks.append(f"Subject: {subject}\n\n{body}")
            return "\n\n---\n\n".join(chunks)
        if kind == "document":
            return artifact.get("content", "") or ""
        if kind == "slides":
            chunks = []
            for slide in artifact.get("slides", []):
                title = slide.get("title", "")
                content = slide.get("content", "")
                notes = slide.get("speaker_notes", "")
                chunks.append(
                    f"Slide {slide.get('slide_number', '')}: {title}\n"
                    f"{content}\n"
                    f"Speaker notes: {notes}"
                )
            return "\n\n".join(chunks)
        return ""

    def _project_hints_for_artifact(self, artifact: dict) -> str:
        candidate_ids: list[str] = []
        related = artifact.get("related_project") or artifact.get("related_project_id")
        if related:
            candidate_ids.append(related)
        for proj_update in artifact.get("project_updates", []) or []:
            pid = proj_update.get("project_id") if isinstance(proj_update, dict) else None
            if pid:
                candidate_ids.append(pid)

        seen: set[str] = set()
        hint_lines: list[str] = []
        for pid in candidate_ids:
            if pid in seen or pid not in self.projects:
                continue
            seen.add(pid)
            project = self.projects[pid]
            term_strs: list[str] = []
            for t in project.specialized_terms:
                if t.expansion:
                    term_strs.append(f"{t.surface} ({t.category}, expands to: {t.expansion})")
                else:
                    term_strs.append(f"{t.surface} ({t.category})")
            if term_strs:
                hint_lines.append(
                    f"- Project {project.name}: {', '.join(term_strs)}"
                )

        if not hint_lines:
            return "Project vocabulary hints: (none available for this artifact)"
        return "Project vocabulary hints (use to disambiguate category, not as a filter):\n" + "\n".join(hint_lines)

    async def _extract_jargon_for_artifact(
        self, artifact_id: str, artifact: dict, text: str,
    ) -> dict:
        truncated_text = text[: self._JARGON_TEXT_LIMIT]
        prompt = (
            self._jargon_prompt
            .replace("{{ARTIFACT_TYPE}}", artifact.get("type", ""))
            .replace("{{ARTIFACT_TITLE}}", artifact.get("title") or artifact.get("subject") or artifact_id)
            .replace("{{PROJECT_HINTS}}", self._project_hints_for_artifact(artifact))
            .replace("{{ARTIFACT_TEXT}}", truncated_text)
        )
        raw = await self._call_llm(prompt, max_tokens=4000)
        try:
            parsed = extract_json(raw)
        except Exception as e:
            logger.warning("Could not parse jargon JSON for %s: %s", artifact_id, e)
            return {"surface_terms": [], "concept_clusters": []}
        if not isinstance(parsed, dict):
            return {"surface_terms": [], "concept_clusters": []}
        response = JargonExtractionResponse.model_validate(parsed)
        return response.model_dump(mode="json")

    def _write_workspace_jargon_aggregate(
        self, jargon_by_artifact: dict[str, dict]
    ) -> None:
        surface_totals: dict[str, int] = {}
        cluster_index: dict[str, dict] = {}

        for artifact_id, payload in jargon_by_artifact.items():
            for term in payload.get("surface_terms", []) or []:
                surface = (term.get("surface") or "").strip()
                if not surface:
                    continue
                occ = int(term.get("occurrences", 1) or 1)
                surface_totals[surface] = surface_totals.get(surface, 0) + occ
            for cluster in payload.get("concept_clusters", []) or []:
                canonical = (cluster.get("canonical") or "").strip()
                if not canonical:
                    continue
                entry = cluster_index.setdefault(
                    canonical,
                    {
                        "canonical": canonical,
                        "variants": set(),
                        "category": cluster.get("category", "industry_term"),
                        "artifact_ids": set(),
                    },
                )
                for v in cluster.get("variants", []) or []:
                    if isinstance(v, str) and v.strip():
                        entry["variants"].add(v.strip())
                entry["artifact_ids"].add(artifact_id)

        aggregate = {
            "surface_terms": [
                {"surface": s, "occurrences": surface_totals[s]}
                for s in sorted(surface_totals, key=lambda k: (-surface_totals[k], k))
            ],
            "concept_clusters": [
                {
                    "canonical": entry["canonical"],
                    "variants": sorted(entry["variants"]),
                    "category": entry["category"],
                    "artifact_ids": sorted(entry["artifact_ids"]),
                }
                for entry in sorted(cluster_index.values(), key=lambda e: e["canonical"].lower())
            ],
        }
        fileio.write_json(self.output_dir / "workspace_jargon.json", aggregate)

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------

    def _run_consistency_checks(self) -> dict:
        issues: list[dict] = []

        # Check follow-up emails reference valid meetings
        for aid, art in self.artifacts.items():
            if art.get("type") != "email_thread":
                continue
            # Check if any dependent meeting exists
            for dep_id in art.get("depends_on", []):
                if dep_id in self.artifacts:
                    dep = self.artifacts[dep_id]
                    if dep.get("type") == "meeting" and dep.get("realized_status") in ("skipped", "canceled"):
                        issues.append({
                            "severity": "warning",
                            "type": "followup_to_canceled_meeting",
                            "message": f"Email thread {aid} follows up on {dep.get('realized_status')} meeting {dep_id}",
                            "artifact_ids": [aid, dep_id],
                        })

        # Check documents reference valid meetings
        for aid, art in self.artifacts.items():
            if art.get("type") != "document":
                continue
            related = art.get("related_meeting_id")
            if related and related in self.artifacts:
                meeting = self.artifacts[related]
                if meeting.get("realized_status") in ("skipped", "canceled"):
                    issues.append({
                        "severity": "warning",
                        "type": "doc_references_canceled_meeting",
                        "message": f"Document {aid} references {meeting.get('realized_status')} meeting {related}",
                        "artifact_ids": [aid, related],
                    })

        # Check action items assigned to valid people
        for ai in self.final_state.action_items.values():
            if ai.assignee_id and not self.org.has_person(ai.assignee_id):
                issues.append({
                    "severity": "error",
                    "type": "invalid_assignee",
                    "message": f"Action item '{ai.description}' assigned to unknown person {ai.assignee_id}",
                    "artifact_ids": [ai.source_meeting_id or ""],
                })

        return {
            "total_issues": len(issues),
            "errors": len([i for i in issues if i["severity"] == "error"]),
            "warnings": len([i for i in issues if i["severity"] == "warning"]),
            "issues": issues,
        }

    def _write_consistency_markdown(self, report: dict) -> None:
        lines = [
            "# Consistency Report",
            "",
            f"Total issues: {report['total_issues']}",
            f"Errors: {report['errors']}",
            f"Warnings: {report['warnings']}",
            "",
        ]
        for issue in report.get("issues", []):
            severity = issue["severity"].upper()
            lines.append(f"- **[{severity}]** {issue['message']}")

        path = self.output_dir / "consistency_report.md"
        fileio.write_text(path, "\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # LLM API
    # ------------------------------------------------------------------

    async def _call_llm(self, user_prompt: str, *, max_tokens: int = 1000) -> str:
        from ..utils.usage_tracker import tracker

        # Step 6 calls are short extractions with tight budgets; at the API's
        # default (medium) effort, reasoning models can spend the whole budget
        # on hidden reasoning and return empty content.
        extra = (
            {"reasoning_effort": "low"}
            if model_uses_reasoning(getattr(self.llm, "model", ""))
            else {}
        )
        response = await self.llm.create_message(
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            **extra,
        )
        tracker.track_llm_response(response)
        if response.truncated:
            logger.warning(
                "Step 6 LLM call truncated by provider (stop_reason=%s, "
                "output_tokens=%d, max_tokens=%d).",
                response.stop_reason, response.output_tokens, max_tokens,
            )
        return response.text

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> dict:
        with open(self.output_dir / "action_items.json") as f:
            action_items = json.load(f)
        logger.info("Loaded step 6: %d action items", len(action_items))
        return {"action_items": action_items}
