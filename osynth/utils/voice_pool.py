"""Voice pool management for consistent TTS voice assignment.

The ``VoiceSource`` protocol abstracts where voice references come from
(CommonVoice, custom datasets, future sources). ``VoicePool`` consumes any
source and handles assignment.
"""
from __future__ import annotations

import random
from typing import Protocol

from ..models.people import ExternalPerson, Person


# ---------------------------------------------------------------------------
# Voice source abstraction
# ---------------------------------------------------------------------------


class VoiceSource(Protocol):
    """Interface for pluggable voice reference providers."""

    def list_voices(self, gender: str | None = None) -> list[str]:
        """Return available voice IDs, optionally filtered by gender.

        ``gender`` is one of ``"male"``, ``"female"``, ``"non-binary"``,
        or ``None`` (all).
        """
        ...


class StaticVoiceSource:
    """Simplest source: explicit lists of voice IDs."""

    def __init__(
        self,
        male: list[str] | None = None,
        female: list[str] | None = None,
        nonbinary: list[str] | None = None,
    ):
        self._male = list(male or [])
        self._female = list(female or [])
        self._nonbinary = list(nonbinary or [])

    def list_voices(self, gender: str | None = None) -> list[str]:
        if gender == "male":
            return list(self._male)
        if gender == "female":
            return list(self._female)
        if gender == "non-binary":
            return list(self._nonbinary) if self._nonbinary else list(self._male) + list(self._female)
        return list(self._male) + list(self._female) + list(self._nonbinary)


class CatalogVoiceSource:
    """Load voice references from a pre-cached voice catalog directory.

    Works with any dataset (CommonVoice, CORAA, etc.) that has been
    prepared into the standard catalog format.

    Supports two layouts:

    1. **Catalog-based** (preferred): ``voice_catalog.json`` + ``samples/`` dir::

           cache_dir/
             voice_catalog.json
             samples/
               speaker_id_1.wav
               speaker_id_2.wav

    2. **Gender-split**: ``male/`` and ``female/`` subdirectories::

           cache_dir/
             male/
               speaker_id_1.wav
             female/
               speaker_id_2.wav
    """

    def __init__(self, cache_dir: str, *, screened_only: bool = True):
        from pathlib import Path

        self._cache = Path(cache_dir)
        self._catalog: dict | None = None
        self._screened_only = screened_only

    def _load_catalog(self) -> dict | None:
        if self._catalog is not None:
            return self._catalog
        import json

        catalog_path = self._cache / "voice_catalog.json"
        if catalog_path.exists():
            with open(catalog_path, encoding="utf-8") as f:
                self._catalog = json.load(f)
        return self._catalog

    def list_voices(self, gender: str | None = None) -> list[str]:
        catalog = self._load_catalog()
        if catalog and "speakers" in catalog:
            return self._list_from_catalog(catalog, gender)
        return self._list_from_dirs(gender)

    def _list_from_catalog(self, catalog: dict, gender: str | None) -> list[str]:
        target_genders: set[str] = set()
        if gender is None:
            target_genders = {"male", "female", "non-binary", "unknown"}
        elif gender == "non-binary":
            # Prefer voices labelled non-binary; fall back to the full pool
            # when the catalog has none.
            has_nonbinary = any(
                meta.get("gender") == "non-binary"
                for meta in catalog["speakers"].values()
            )
            target_genders = (
                {"non-binary"} if has_nonbinary
                else {"male", "female", "unknown"}
            )
        elif gender in ("male", "female"):
            target_genders = {gender}

        screened_catalog = "screening_version" in catalog

        def _matches(meta: dict) -> bool:
            return not target_genders or meta.get("gender") in target_genders

        def _rank(meta: dict) -> tuple[int, int, int]:
            # Lower is better. Sort by final_phone_class (strong > acceptable),
            # then by duration band (preferred > soft > fallback), then by
            # trailing silence ascending.
            class_order = {"strong": 0, "acceptable": 1, "weak": 2, "unknown": 3}
            band_order = {
                "preferred": 0,
                "soft_short": 1,
                "soft_long": 1,
                "fallback": 2,
                "rejected": 3,
            }
            return (
                class_order.get(meta.get("final_phone_class"), 4),
                band_order.get(meta.get("duration_band"), 4),
                int(meta.get("trailing_silence_ms", 1_000_000)),
            )

        # Primary pool: screened-passed voices.
        primary: list[tuple[tuple[int, int, int], str]] = []
        fallback: list[str] = []
        for speaker_id, meta in catalog["speakers"].items():
            if not _matches(meta):
                continue
            if screened_catalog and self._screened_only:
                if meta.get("screening_passed"):
                    primary.append((_rank(meta), speaker_id))
                # else: drop entirely when screened_only
            else:
                # Old (unscreened) catalogs, or screened_only disabled: include all.
                if meta.get("screening_passed") is False:
                    fallback.append(speaker_id)
                else:
                    primary.append((_rank(meta), speaker_id))

        primary.sort()
        return [sid for _, sid in primary] + sorted(fallback)

    def _list_from_dirs(self, gender: str | None) -> list[str]:
        voices: list[str] = []
        if gender == "non-binary":
            genders = ["male", "female"]
        elif gender in ("male", "female"):
            genders = [gender]
        else:
            genders = ["male", "female"]
        for g in genders:
            d = self._cache / g
            if d.is_dir():
                voices.extend(p.stem for p in sorted(d.glob("*.wav")))
        return voices


# Backward-compatible alias
CommonVoiceSource = CatalogVoiceSource


# ---------------------------------------------------------------------------
# Voice pool
# ---------------------------------------------------------------------------


class VoicePool:
    """Manages TTS voice assignments for workspace people."""

    def __init__(self, source: VoiceSource, seed: int | None = None):
        self._source = source
        self._assignments: dict[str, str] = {}
        self._rng = random.Random(seed)

    def assign_all(self, people: dict[str, Person | ExternalPerson]) -> dict[str, str]:
        """Assign voices to all people. Returns person_id -> voice_id mapping."""
        male_people = [p for p in people.values() if p.gender == "male"]
        female_people = [p for p in people.values() if p.gender == "female"]
        nonbinary_people = [p for p in people.values() if p.gender == "non-binary"]

        self._rng.shuffle(male_people)
        self._rng.shuffle(female_people)
        self._rng.shuffle(nonbinary_people)

        male_pool = self._source.list_voices("male")
        female_pool = self._source.list_voices("female")
        nonbinary_pool = self._source.list_voices("non-binary")
        all_pool = self._source.list_voices()

        if not all_pool:
            raise RuntimeError("No voices available from the configured source")

        for i, person in enumerate(male_people):
            pool = male_pool or all_pool
            self._assignments[person.id] = pool[i % len(pool)]

        for i, person in enumerate(female_people):
            pool = female_pool or all_pool
            self._assignments[person.id] = pool[i % len(pool)]

        nb_pool = nonbinary_pool or all_pool
        self._rng.shuffle(nb_pool)
        for i, person in enumerate(nonbinary_people):
            self._assignments[person.id] = nb_pool[i % len(nb_pool)]

        return dict(self._assignments)

    def get_voice(self, person_id: str) -> str | None:
        return self._assignments.get(person_id)

    def get_voice_by_name(
        self, name: str, people: dict[str, Person | ExternalPerson]
    ) -> str | None:
        for pid, person in people.items():
            if person.name == name:
                return self._assignments.get(pid)
        return None

    def get_name_to_voice_map(
        self, people: dict[str, Person | ExternalPerson]
    ) -> dict[str, str]:
        return {
            people[pid].name: voice
            for pid, voice in self._assignments.items()
            if pid in people
        }

    def load_assignments(self, assignments: dict[str, str]) -> None:
        self._assignments = dict(assignments)

    def to_dict(self) -> dict[str, str]:
        return dict(self._assignments)
