"""Normalize AMI manual transcripts and synthetic formatted turns to a shared plain-text surface.

Both sides produce a single string of ``Speaker N: <text>`` lines. Consecutive
turns by the same speaker are merged into a single line so the discriminator
cannot exploit turn-segmentation cadence as a free signal.
Allowed surface tokens after normalization:

* words (any orthography),
* filler tokens ``uh / um / er / hmm / mm-hmm``,
* trailing-hyphen truncation fragments (``Ye-``, ``go-``),
* sentence punctuation ``. , ? !`` (attached to the preceding word, no leading space),
* the ``Speaker N:`` line prefix.

Anything else (vocal sounds, transcriber meta tokens, em-dashes, real names, ``(??)``,
``(word?)``) is dropped on both sides so the discriminator does not see a free signal.

The §1.5 audit of the AMI Meeting Corpus manual annotations (v1.6.2, ``words/*.words.xml``)
confirmed AMI's manual orthography carries no ``(??)`` whole-region marker and no
``(word?)`` transcriber-uncertain convention — only ``<w trunc="true">`` reaches the
surface (rendered here as a single trailing hyphen).
"""
from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

NITE_NS = "http://nite.sourceforge.net/"
_ID_ATTR = f"{{{NITE_NS}}}id"
_CHILD_TAG = f"{{{NITE_NS}}}child"

# Tokens we keep when rendering an AMI <w> element. Everything else is dropped.
# `mispronounced="true"` is a pronunciation flag with no orthographic surface.
_DROPPED_AMI_ELEMENTS: frozenset[str] = frozenset(
    {"vocalsound", "nonvocalsound", "gap", "pause", "transformerror", "disfmarker"},
)

# Regex for #id(X)..id(Y) ranges inside <nite:child href="...">.
_RANGE_RE = re.compile(r"id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


# ---------------------------------------------------------------------------
# AMI side
# ---------------------------------------------------------------------------


def _parse_words(words_path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return `(ordered_ids, word_entries)` for one channel.

    `ordered_ids` preserves the original document order, including IDs that
    belong to dropped elements (`<vocalsound>`, `<disfmarker>`, etc.) — this
    keeps segment range walks correct when a segment spans a dropped element.
    `word_entries` only contains real `<w>` tokens; lookups for dropped IDs
    miss the dict and are skipped at render time.
    """
    tree = ET.parse(words_path)
    root = tree.getroot()
    ordered_ids: list[str] = []
    entries: dict[str, dict[str, Any]] = {}
    for child in root:
        wid = child.get(_ID_ATTR)
        if not wid:
            continue
        ordered_ids.append(wid)
        local = child.tag.split("}", 1)[-1]
        if local in _DROPPED_AMI_ELEMENTS:
            continue
        if local != "w":
            continue
        text = (child.text or "").strip()
        if not text:
            continue
        is_punc = child.get("punc") == "true"
        is_trunc = child.get("trunc") == "true"
        try:
            start = float(child.get("starttime") or 0.0)
        except ValueError:
            start = 0.0
        if is_trunc:
            text = text + "-"
        entries[wid] = {
            "text": text,
            "punc": is_punc,
            "trunc": is_trunc,
            "start": start,
        }
    return ordered_ids, entries


def _parse_segments(segments_path: Path) -> list[tuple[float, list[tuple[str, str | None]]]]:
    """Return [(start_time, [(first_id, last_id), ...]), ...] sorted by start_time.

    Each segment's `<nite:child href="...">` element points at a word-id range.
    A segment may contain multiple ranges (rare but legal).
    """
    tree = ET.parse(segments_path)
    root = tree.getroot()
    segments: list[tuple[float, list[tuple[str, str | None]]]] = []
    # `<segment>` lives in the default (no) namespace inside the nite root.
    for seg in root.findall("segment"):
        try:
            start = float(seg.get("transcriber_start") or 0.0)
        except ValueError:
            start = 0.0
        ranges: list[tuple[str, str | None]] = []
        for child in seg.findall(_CHILD_TAG):
            href = child.get("href") or ""
            for first, last in _RANGE_RE.findall(href):
                ranges.append((first, last))
        if ranges:
            segments.append((start, ranges))
    segments.sort(key=lambda s: s[0])
    return segments


def _render_segment_text(
    ordered_ids: list[str],
    words: dict[str, dict[str, Any]],
    ranges: list[tuple[str, str | None]],
) -> str:
    """Walk an AMI segment's word-id ranges and join into a single line.

    Punctuation tokens attach to the preceding word; truncation tokens already
    carry their trailing hyphen from `_parse_words`. IDs from dropped element
    types (`<vocalsound>`, `<disfmarker>`, etc.) are absent from `words` and
    are silently skipped while walking the document-order range.
    """
    id_index = {wid: i for i, wid in enumerate(ordered_ids)}
    pieces: list[str] = []
    for first, last in ranges:
        if first not in id_index:
            continue
        first_i = id_index[first]
        last_i = id_index[last] if last and last in id_index else first_i
        for i in range(first_i, last_i + 1):
            wid = ordered_ids[i]
            entry = words.get(wid)
            if entry is None:
                continue
            text = entry["text"]
            if entry["punc"] and pieces:
                pieces[-1] = pieces[-1] + text
            else:
                pieces.append(text)
    line = " ".join(pieces)
    # Collapse double spaces that can arise from unusual XML whitespace.
    line = re.sub(r"\s+", " ", line).strip()
    return line


def normalize_ami_meeting(
    ami_root: str | Path,
    meeting_id: str,
) -> str:
    """Render a single AMI meeting as `Speaker N: ...` lines in turn-start order.

    `ami_root` should point at `ami_public_manual_1.6.2/` (the directory containing
    `words/` and `segments/`). Channel letters in the AMI files (`A`, `B`, ...) are
    renamed to `Speaker 1`, `Speaker 2`, ... in first-appearance order so synthetic
    and AMI transcripts share the same label space.
    """
    root = Path(ami_root)
    words_dir = root / "words"
    segments_dir = root / "segments"

    # Find all channel letters for this meeting by globbing the segments dir.
    channel_files = sorted(segments_dir.glob(f"{meeting_id}.*.segments.xml"))
    if not channel_files:
        raise FileNotFoundError(
            f"No segments files found for AMI meeting {meeting_id!r} under {segments_dir}",
        )

    # Build per-channel word table.
    by_channel_ordered: dict[str, list[str]] = {}
    by_channel_words: dict[str, dict[str, dict[str, Any]]] = {}
    by_channel_segments: dict[str, list[tuple[float, list[tuple[str, str | None]]]]] = {}
    for seg_path in channel_files:
        # filename: {meeting}.{channel}.segments.xml
        stem = seg_path.name
        # Strip the trailing `.segments.xml` then split the meeting id off the channel letter.
        base = stem[: -len(".segments.xml")]
        channel = base[len(meeting_id) + 1 :]
        word_path = words_dir / f"{meeting_id}.{channel}.words.xml"
        if not word_path.exists():
            logger.warning("words file missing for %s.%s", meeting_id, channel)
            continue
        ordered, entries = _parse_words(word_path)
        by_channel_ordered[channel] = ordered
        by_channel_words[channel] = entries
        by_channel_segments[channel] = _parse_segments(seg_path)

    # Build a flat list of (start_time, channel, segment_ranges) and sort.
    flat: list[tuple[float, str, list[tuple[str, str | None]]]] = []
    for channel, segs in by_channel_segments.items():
        for start, ranges in segs:
            flat.append((start, channel, ranges))
    flat.sort(key=lambda r: (r[0], r[1]))

    # Assign Speaker N labels in first-appearance order.
    channel_to_label: dict[str, str] = {}
    rows: list[tuple[str, str]] = []
    for start, channel, ranges in flat:
        if channel not in channel_to_label:
            channel_to_label[channel] = f"Speaker {len(channel_to_label) + 1}"
        text = _render_segment_text(
            by_channel_ordered[channel],
            by_channel_words[channel],
            ranges,
        )
        if not text:
            continue
        rows.append((channel_to_label[channel], text))
    return _format_merged_rows(rows)


def list_ami_meetings(ami_root: str | Path) -> list[str]:
    """List meeting IDs available under `ami_root/segments/` by stripping suffixes."""
    root = Path(ami_root) / "segments"
    if not root.exists():
        return []
    seen: set[str] = set()
    for path in root.glob("*.segments.xml"):
        # Filename like EN2001a.A.segments.xml — meeting id is everything before the
        # last two dot-separated suffixes (channel + "segments.xml").
        stem = path.name[: -len(".segments.xml")]
        # Strip the channel letter (single-letter suffix after the meeting id).
        if "." in stem:
            meeting_id, _ = stem.rsplit(".", 1)
            seen.add(meeting_id)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Style-vs-eval split
# ---------------------------------------------------------------------------
#
# The realism rewrite (step 4) shows the LLM a real AMI excerpt as a style
# anchor. Any downstream real-vs-synthetic discriminability evaluation then
# asks an LLM judge to tell real from synthetic — also against AMI. If the same AMI
# meetings are visible to both, the metric is leaky: any artefact the
# rewrite copied verbatim from its style anchor would also surface in the
# discriminator's "real" sample, so a low accuracy score could just mean
# the rewrite memorized that one meeting. The split below partitions the
# normalized AMI corpus into disjoint `style_ref` and `discriminator_eval`
# sets via a stable hash, so both callers see the same partition without
# any on-disk state.

_DEFAULT_SPLIT_SEED = "ami_realism_split_v1"


def _hash_partition(
    meeting_ids: list[str],
    *,
    style_fraction: float,
    salt: str,
) -> tuple[list[str], list[str]]:
    style: list[str] = []
    evaluation: list[str] = []
    for meeting_id in sorted(meeting_ids):
        digest = hashlib.sha1(f"{salt}::{meeting_id}".encode("utf-8")).digest()
        rank = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if rank < style_fraction:
            style.append(meeting_id)
        else:
            evaluation.append(meeting_id)
    return style, evaluation


def _read_id_list(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
    return out


def load_ami_split(
    normalized_dir: str | Path,
    *,
    style_fraction: float = 0.5,
    salt: str = _DEFAULT_SPLIT_SEED,
) -> tuple[list[str], list[str]]:
    """Return (style_ref, discriminator_eval) for the AMI corpus.

    Lookup order:
      1. On-disk manifest at `<normalized_dir>/splits/<salt>/style_ref.txt`
         and `<normalized_dir>/splits/<salt>/discriminator_eval.txt` — one
         meeting id per line, `#` lines ignored. This is the source of truth
         when present.
      2. Hash partition over the `*.txt` files in `normalized_dir`.

    The hash partition is `sha1(salt + meeting_id)` mapped to the unit
    interval and bucketed at `style_fraction`; it is deterministic given
    the same inputs. Persist it with `write_ami_split` so the concrete
    split is visible on disk.
    """
    base = Path(normalized_dir)
    splits_dir = base / "splits" / salt
    style_path = splits_dir / "style_ref.txt"
    eval_path = splits_dir / "discriminator_eval.txt"
    if style_path.exists() and eval_path.exists():
        return _read_id_list(style_path), _read_id_list(eval_path)
    if not base.exists():
        return [], []
    meeting_ids = sorted(p.stem for p in base.glob("*.txt"))
    return _hash_partition(
        meeting_ids, style_fraction=style_fraction, salt=salt,
    )


def write_ami_split(
    normalized_dir: str | Path,
    *,
    meeting_ids: list[str],
    style_fraction: float = 0.5,
    salt: str = _DEFAULT_SPLIT_SEED,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Materialize the deterministic split as two on-disk text files.

    Writes `<normalized_dir>/splits/<salt>/{style_ref,discriminator_eval}.txt`,
    one meeting id per line, sorted. Returns the two paths.
    """
    style, evaluation = _hash_partition(
        meeting_ids, style_fraction=style_fraction, salt=salt,
    )
    base = Path(normalized_dir)
    splits_dir = base / "splits" / salt
    splits_dir.mkdir(parents=True, exist_ok=True)
    style_path = splits_dir / "style_ref.txt"
    eval_path = splits_dir / "discriminator_eval.txt"
    if (style_path.exists() or eval_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing AMI split at {splits_dir}; "
            "pass overwrite=True to replace.",
        )
    header = (
        f"# AMI split — salt={salt}, style_fraction={style_fraction}\n"
        "# One meeting id per line. Source of truth when present.\n"
        "# Regenerate via osynth.tts.ami_normalize.write_ami_split.\n"
    )
    style_path.write_text(header + "\n".join(style) + "\n", encoding="utf-8")
    eval_path.write_text(header + "\n".join(evaluation) + "\n", encoding="utf-8")
    return style_path, eval_path


# ---------------------------------------------------------------------------
# Synthetic side
# ---------------------------------------------------------------------------


_WORD_TRUNC_DOUBLE_RE = re.compile(r"--(?=\s|$|[.,!?])")


def _collapse_double_hyphens(text: str, edits: Iterable[dict[str, Any]]) -> str:
    """Collapse the TTS-side `aparata--` form to the read-side `aparata-`.

    The plan §2.1 mandates that the collapse is driven off the edit layer (so we
    only touch tokens that were *actually* emitted as `word_truncation`). We scan
    `semantic_edit_layer` for single-hyphen truncation fragments, locate the
    corresponding double-hyphen TTS fragment, and collapse one hyphen. Falls back
    to a regex that targets word-final `--` only when an edit layer entry is
    missing.
    """
    spans: list[tuple[int, int]] = []
    for edit in edits or []:
        if not isinstance(edit, dict):
            continue
        if edit.get("type") != "word_truncation":
            continue
        fragment = str(edit.get("surface") or "").strip()
        if not fragment.endswith("-"):
            continue
        tts_fragment = fragment[:-1] + "--"
        try:
            preferred = int(edit.get("span_start", 0))
        except (TypeError, ValueError):
            preferred = 0
        start = _find_near(text, tts_fragment, preferred)
        if start >= 0:
            spans.append((start, start + len(tts_fragment)))

    if spans:
        # Apply right-to-left so earlier offsets stay valid.
        spans.sort(reverse=True)
        for start, end in spans:
            piece = text[start:end]
            collapsed = piece[:-2] + "-" if piece.endswith("--") else piece
            text = text[:start] + collapsed + text[end:]
        return text

    # No edit-layer entries: fall back to word-final `--` (rare; only fires when
    # someone hand-authored a tts_text without an accompanying edit-layer entry).
    return _WORD_TRUNC_DOUBLE_RE.sub("-", text)


def _find_near(text: str, needle: str, preferred: int) -> int:
    """Find ``needle`` close to ``preferred`` when possible, else first match."""
    if not needle:
        return -1
    window_start = max(0, preferred - 24)
    window_end = min(len(text), preferred + len(needle) + 24)
    local = text.find(needle, window_start, window_end)
    if local >= 0:
        return local
    return text.find(needle)


def _strip_meta_noise(text: str) -> str:
    """Drop bracketed meta markers that have no AMI parity.

    Synthetic-side `tts_text` should not carry meta markers in practice (those
    live in the edit layer), but this is defensive: anything in [brackets],
    (parentheses) like `(??)`, or `<angle-tags>` is stripped so we never feed a
    free signal to the discriminator.
    """
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\(\?\?\)", "", text)
    # `(word?)` style transcriber uncertainty.
    text = re.sub(r"\(\w+\?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_appearance_label_map(turns: list[dict[str, Any]]) -> dict[str, str]:
    """Map every distinct speaker_id (or speaker name fallback) to `Speaker N`."""
    mapping: dict[str, str] = {}
    for turn in turns:
        key = str(turn.get("speaker_id") or turn.get("speaker") or "")
        if not key or key in mapping:
            continue
        mapping[key] = f"Speaker {len(mapping) + 1}"
    return mapping


def normalize_synthetic_turns(turns: list[dict[str, Any]]) -> str:
    """Render `formatted_turns.json` content as the same `Speaker N: ...` surface.

    Reads `tts_text` per turn (the only surface that already carries fillers,
    repetitions, false starts, prolongations, and truncations). Collapses the
    TTS-side `--` truncation rendering back to a single `-` for parity with
    the AMI side. Strips any synthetic-only meta markers. Consecutive turns by
    the same speaker are merged into a single line.
    """
    labels = _first_appearance_label_map(turns)
    rows: list[tuple[str, str]] = []
    for turn in turns:
        key = str(turn.get("speaker_id") or turn.get("speaker") or "")
        if not key:
            continue
        speaker_label = labels.get(key) or f"Speaker {len(labels) + 1}"
        text = str(turn.get("tts_text") or turn.get("text") or "")
        text = _collapse_double_hyphens(text, turn.get("semantic_edit_layer") or [])
        text = _strip_meta_noise(text)
        if not text:
            continue
        rows.append((speaker_label, text))
    return _format_merged_rows(rows)


def _format_merged_rows(rows: list[tuple[str, str]]) -> str:
    """Render `(speaker_label, text)` rows, merging consecutive same-speaker rows.

    Adjacent rows from the same speaker join into a single `Speaker N: a b`
    line with a single space between the segments — equivalent to how a reader
    would collapse back-to-back utterances from the same person into one block.
    """
    merged: list[tuple[str, list[str]]] = []
    for label, text in rows:
        if not text:
            continue
        if merged and merged[-1][0] == label:
            merged[-1][1].append(text)
        else:
            merged.append((label, [text]))
    return "\n".join(f"{label}: {' '.join(parts)}" for label, parts in merged)


# ---------------------------------------------------------------------------
# Parity guarantee
# ---------------------------------------------------------------------------


# After normalization, only these characters may appear in the surface beyond words
# and the standard ASCII alphanumeric range. Used by tests to assert parity.
_DISALLOWED_PARITY_RE = re.compile(
    r"<[^>]+>"            # angle-bracket meta
    r"|\[[^\]]*\]"        # square-bracket meta
    r"|\(\?\?\)"          # AMI-style whole-region unclear
    r"|\(\w+\?\)"         # transcriber-uncertain word
    r"|—"                  # em-dash
    r"|--",                # double-hyphen truncation (must be collapsed)
)


def parity_violations(text: str) -> list[str]:
    """Return a list of disallowed markers found in `text`, empty if surface is clean."""
    return _DISALLOWED_PARITY_RE.findall(text)
