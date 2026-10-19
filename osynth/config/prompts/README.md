# Prompt Templates

This directory contains the prompt templates used by the synthetic data pipeline. Templates are organized by artifact type and become LLM input after variable substitution — do not add comments or non-prompt text to the `.txt` files.

Variables use `{{VARIABLE_NAME}}` syntax and are substituted by the pipeline step before sending to the LLM.
Language markers such as `{lang:name}` are resolved first using `osynth/utils/language.py`.

## Generation Pattern

The pipeline uses a **sketch-then-expand** pattern for parallel generation:

1. **Sketch** — A single LLM call produces a lightweight outline (e.g., org sketch with person stubs)
2. **Detail** — Parallel LLM calls expand each sketch item into a full artifact (e.g., person detail)

This enables concurrent generation while maintaining coherence through shared context.

---

## workspace_sampling/ (Step 1: Workspace Foundation)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `system.txt` | System prompt for all Step 1 calls. Sets the LLM's role as a workplace simulation data generator. | — |
| `company.txt` | Generate company profile (domain, industry, size, location) | `COMPANY_SIZE`, `INDUSTRY`, `DOMAIN`, `COMPANY_SIZE_ENUM` |
| `org_spine.txt` | **Frame**: Team structure, manager sketches, reporting tree, per-team headcount + role mix, optional externals | `COMPANY_JSON`, `NUM_PEOPLE`, `ORG_CONSTRAINTS` |
| `team_people_sketch.txt` | **Fanout**: IC sketches for one team, conditioned on the spine | `COMPANY_SUMMARY`, `ORG_SPINE_SUMMARY`, `TEAM_JSON`, `TEAM_ID`, `RESERVED_IDS`, `ID_RANGE`, `ID_RANGE_FIRST` |
| `person_detail.txt` | **Expand**: Full person profile — role, personality, expertise, speaking style | `COMPANY_SUMMARY`, `ORG_SKETCH_SUMMARY`, `PERSON_SKETCH_JSON`, `PERSON_ID` |
| `external_person_detail.txt` | **Expand**: External contact profile (clients, vendors, partners) | `COMPANY_SUMMARY`, `PERSON_SKETCH_JSON`, `RELATIONSHIP_TYPE`, `PERSON_ID` |
| `project_index.txt` | **Frame**: Group projects into 1-4 clusters with reserved ID spans | `COMPANY_JSON`, `ORG_SUMMARY`, `START_DATE`, `END_DATE`, `NUM_PROJECTS` |
| `project_cluster_sketch.txt` | **Fanout**: Project sketches for one cluster | `COMPANY_JSON`, `ORG_SUMMARY`, `START_DATE`, `END_DATE`, `CLUSTER_JSON` |
| `project_detail.txt` | **Expand**: Full project with timeline, milestones, blockers | `COMPANY_SUMMARY`, `ORG_SUMMARY`, `START_DATE`, `END_DATE`, `PROJECT_SKETCH_JSON`, `PROJECT_ID` |
| `recurring_meetings_company.txt` | Generate company-wide recurring meetings (all-hands, exec syncs) | `COMPANY_JSON`, `ORG_SUMMARY`, `PROJECTS_SUMMARY`, `WORKING_HOURS_START`, `WORKING_HOURS_END`, `TIMEZONE` |
| `recurring_meetings_team.txt` | Generate one team's local rituals (1x1s, standups, team syncs) | shared meeting vars + `TEAM_JSON`, `TEAM_ID` |
| `recurring_meetings_cross.txt` | Generate cross-functional + external recurring meetings | shared meeting vars |

**Flow**: `company` → `org_spine` → `team_people_sketch` (parallel per team) → `person_detail` (parallel) + `external_person_detail` (parallel) → `project_index` → `project_cluster_sketch` (parallel per cluster) → `project_detail` (parallel) → `recurring_meetings_company` ‖ `recurring_meetings_team` (parallel per team) ‖ `recurring_meetings_cross`

## time_period/ (Step 2: Time Period Planning)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `system.txt` | System prompt for narrative generation. Sets the LLM's role as a workplace narrative designer. | — |
| `narrative.txt` | Generate narrative arc over the full time period — major events, project trajectories, team dynamics | `COMPANY_JSON`, `ORG_SUMMARY`, `PROJECTS_JSON`, `START_DATE`, `END_DATE`, `NUM_WEEKS` |
| `week_frame.txt` | **Frame**: Authoritative weekly summary, forward brief, state changes, action-items resolved, and per-family content targets | `COMPANY_SUMMARY`, `NARRATIVE_SUMMARY`, `WEEK_START`, `WEEK_END`, `WEEK_NUM`, `TOTAL_WEEKS`, `WORKING_DAYS_LIST`, `STATE_SUMMARY`, `EVENTS_THIS_WEEK`, `RECURRING_MEETINGS_SUMMARY`, `PREV_FORWARD_BRIEF` |
| `week_meetings.txt` | **Fanout**: Adhoc meetings + outcomes for the week, conditioned on the frame | shared week vars + `PEOPLE_ROSTER`, `PROJECTS_ROSTER`, `WEEK_FRAME_JSON` |
| `week_emails.txt` | **Fanout**: Email threads for the week, conditioned on the frame, including active correspondents vs CC recipients | shared week vars + `PEOPLE_ROSTER`, `PROJECTS_ROSTER`, `WEEK_FRAME_JSON` |
| `week_documents.txt` | **Fanout**: Documents authored this week, conditioned on the frame | shared week vars + `PEOPLE_ROSTER`, `PROJECTS_ROSTER`, `WEEK_FRAME_JSON` |
| `week_slides.txt` | **Fanout**: Slide presentations tied to planned meetings (skipped when slides_target=0) | shared week vars + `WEEK_FRAME_JSON`, `WEEK_MEETINGS_JSON` |

**Flow**: `narrative` (single call) → for each week (sequential): `week_frame` → `week_meetings` ‖ `week_emails` ‖ `week_documents` (parallel) → `week_slides` (only when needed) → typed assembly into one `WeekPlan`.

## meetings/ (Step 4: Artifact Generation)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `system.txt` | System prompt for meeting generation. | — |
| `overview.txt` | **Sketch**: Meeting overview with personas, outline, turn-by-turn discussion flow | `MEETING_TITLE`, `MEETING_TYPE`, `START_TIME`, `END_TIME`, `DURATION_MINUTES`, `ATTENDEES_DETAIL`, `COMPANY_SUMMARY`, `PURPOSE`, `AGENDA`, `PROJECT_SUMMARIES`, `ACTION_ITEMS`, `RECENT_DECISIONS`, `PRIOR_MEETING_SUMMARIES`, `DOCUMENTS_CONTEXT`, `SLIDES_CONTEXT`, `EXPECTED_OUTCOMES` |
| `transcript.txt` | **Expand**: Full transcript as JSON array of turns with natural speech patterns | `MEETING_TITLE`, `MEETING_TYPE`, `DURATION_MINUTES`, `MEETING_OVERVIEW`, `ATTENDEE_ROSTER`, `ATTENDEES_SPEAKING_PROFILES`, `SLIDE_CONTEXT` |
| `event_extraction.txt` | Extract structured events (action items, decisions, project updates) from transcript | `MEETING_TITLE`, `ATTENDEE_ROSTER`, `TRANSCRIPT_TEXT` |
| `anti_exposition.txt` | Optional guidance appended to the transcript prompt when `generation.anti_exposition` is on: technical content is debated, not narrated | — |

**Flow**: `overview` → `transcript` → `event_extraction`. Also used by Step 5 for event reconciliation.

## emails/ (Step 4: Artifact Generation)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `system.txt` | System prompt for email generation. | — |
| `thread.txt` | Generate next email message in a thread — tone, conventions, signatures | `COMPANY_SUMMARY`, `THREAD_HISTORY`, `SENDER_ID`, `SENDER_NAME`, `SENDER_PROFILE`, `RECIPIENT_IDS`, `RECIPIENT_NAMES`, `CC_IDS`, `CC_NAMES`, `THREAD_SUBJECT`, `MESSAGE_PURPOSE`, `TIMESTAMP_CONTEXT`, `WORKSPACE_CONTEXT`, `RELATED_CONTEXT`, `MESSAGE_ID`, `THREAD_ID` |

**Flow**: Messages in a thread are generated sequentially, each receiving the prior thread history.

## documents/ (Step 4: Artifact Generation)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `system.txt` | System prompt for document generation. | — |
| `document.txt` | Generate a workplace document (spec, proposal, notes, report, wiki) | `COMPANY_SUMMARY`, `DOCUMENT_TITLE`, `DOC_TYPE`, `AUTHOR_SUMMARY`, `PROJECT_CONTEXT`, `DOCUMENT_PURPOSE`, `SHARED_WITH`, `RELATED_CONTEXT`, `DOCUMENT_ID`, `AUTHOR_ID`, `PROJECT_ID` |

## slides/ (Step 4: Artifact Generation)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `system.txt` | System prompt for slide generation. | — |
| `content.txt` | Generate slide presentation with layouts and speaker notes | `COMPANY_SUMMARY`, `PRESENTATION_TITLE`, `PRESENTER_SUMMARY`, `AUDIENCE_DESCRIPTION`, `PRESENTATION_PURPOSE`, `NUM_SLIDES`, `RELATED_CONTEXT`, `PRESENTATION_ID`, `AUTHOR_ID` |

## transcripts/ (Step 4 realism loop, Step 7 entities)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `realism_rewrite.txt` | Optional rewrite of a synthetic transcript toward real-meeting style, with an AMI excerpt as style reference | `MEETING_TITLE`, `MEETING_TYPE`, `PARTICIPANTS_CONTEXT`, `PROJECT_CONTEXT`, `STATE_CONTEXT`, `TECHNICALITY_LEVEL`, `AMI_REFERENCE`, `SYNTHETIC_TRANSCRIPT`, `LENGTH_BUDGET_CLAUSE` |
| `realism_judge.txt` | Pairwise judge: which of two transcripts is the real meeting | `TRANSCRIPT_A`, `TRANSCRIPT_B` |
| `realism_refine.txt` | Refine the previous rewrite using the judge's reasoning | `PREVIOUS_REWRITE`, `JUDGE_REASONING`, `STRUCTURAL_ISSUES`, `REAL_REFERENCE`, `LENGTH_BUDGET_CLAUSE` |
| `transcript_entities.txt` | Extract named entities for spoken-form rendering | `PEOPLE_CONTEXT`, `PROJECTS_CONTEXT`, `EXTERNAL_CONTEXT`, `LANGUAGE_NAME` |

**Flow**: `realism_rewrite` → (`realism_judge` → `realism_refine`) × `refine_iterations`.

## postprocess/ (Step 6: Postprocess)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `jargon_extraction.txt` | Extract industry terms and internal coinages per artifact | `ARTIFACT_TYPE`, `ARTIFACT_TITLE`, `ARTIFACT_TEXT`, `PROJECT_HINTS` |

## tts/ (Step 7: Speech Prep)

| File | Purpose | Key Variables |
|------|---------|---------------|
| `spoken_form.txt` | System prompt converting clean turns to spoken-form and semantic-verbatim text | — |
| `enrollment_text.txt` | Candidate enrollment scripts for each speaker's voice clip | `SPEAKER_ID`, `SPEAKER_SUMMARY`, `SOURCE_TRANSCRIPT`, `REPRESENTATIVE_UTTERANCES`, `AUTHORED_SNIPPETS`, `CANDIDATE_COUNT`, `CANDIDATE_PLAN_JSON`, `TARGET_SECONDS`, `MIN_SECONDS`, `MAX_SECONDS` |
| `overlap_annotator.txt` | Choose turns that start in overlap with the previous speaker | `MEETING_MODE`, `TURN_COUNT`, `TARGET_PER_TEN`, `FULL_OVERLAP_PCT`, `TURNS_BLOCK` |
| `backchannel_classifier.txt` | Classify short turns as backchannels | `SPEAKER_PRIORS`, `TURNS_BLOCK` |
| `backchannel_anchor.txt` | Place each backchannel inside the turn it responds to | `TURNS_BLOCK` |

---

## Step-to-Prompt Mapping

| Step | Prompt Directory | # Prompts | Pattern |
|------|-----------------|-----------|---------|
| Step 1 (Workspace) | `workspace_sampling/` | 12 | Frame-then-fanout (company → org spine → team people → person/external detail → project index → cluster sketches → project detail → recurring meetings split into company/team/cross) |
| Step 2 (Time Period) | `time_period/` | 7 | Narrative arc → sequential week frames with per-artifact plans |
| Step 3 (Decomposition) | — | 0 | Computes generation order from Step 2 outputs |
| Step 4 (Artifacts) | `meetings/`, `emails/`, `documents/`, `slides/`, `transcripts/realism_*.txt` | 14 | Overview → transcript → events; optional realism rewrite/judge/refine; sequential threads; parallel docs/slides |
| Step 5 (Events) | `meetings/event_extraction.txt` | (shared) | Reuses meeting event extraction |
| Step 6 (Postprocess) | `postprocess/` | 1 | Jargon extraction; inline prompts for summary/action items/outline |
| Step 7 (Speech Prep) | `tts/`, `transcripts/transcript_entities.txt` | 6 | Spoken-form rewriting, enrollment text, overlap/backchannel annotation, and entity extraction |
| Step 8 (TTS Batch) | — | 0 | Enrollment-first synthesis, no LLM calls |
| Step 9 (Audio Finalize) | — | 0 | Timing, alignment, mixing, and transcript export |
| Step 10 (Room Simulation) | — | 0 | Room acoustics and noise mixing |
| Step 11 (Validation) | — | 0 | Cross-artifact consistency checks |
