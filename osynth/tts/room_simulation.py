"""Conference-room acoustic simulation for in-room speakers.

Applies a synthesized RIR (via pyroomacoustics) to per-speaker WAVs from step 9
and layers configurable background-noise sources (background noise clips,
air-conditioner hum, short sound events, and a small Gaussian
floor) to produce a realistic single-microphone room recording.

This module is purely a numerical / file-IO unit; orchestration lives in
``osynth.steps.step_10_room_simulation``.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..models.calendar import MeetingJoinSpec, ParticipantJoinMode
from ..models.settings import RoomSimulationSettings
from ..utils import fileio

logger = logging.getLogger(__name__)


# Conference-table footprint (centred at room midpoint, x by y, metres).
_TABLE_X = 2.4
_TABLE_Y = 1.0
# Seated-speaker mouth height.
_SPEAKER_HEIGHT_M = 1.2
# Microphone on table.
_MIC_HEIGHT_M = 0.9
# AC unit placement: far corner near ceiling.
_AC_UNIT_OFFSET_M = (0.5, 0.5, 0.3)


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------


@dataclass
class RoomNoisePaths:
    """Resolved paths to the noise corpora used by the room simulator."""

    background_noise_dir: Path | None = None
    hum_noise_dir: Path | None = None
    event_noise_dir: Path | None = None


@dataclass
class RoomSimulationResult:
    combined_room_path: Path
    speaker_room_paths: dict[str, Path] = field(default_factory=dict)
    rt60_measured: float = 0.0
    noise_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def conference_table_positions(
    n: int,
    room_dims: list[float],
    distance_scale: float = 1.0,
) -> list[list[float]]:
    """Return ``n`` 3D speaker positions evenly spaced around an oval table.

    Positions are stable for a given ``n`` and ``room_dims`` so the same
    speakers map to the same chairs across reruns when keyed deterministically.

    ``distance_scale`` uniformly expands the seating radius from the room
    centre (e.g. 1.3 moves every speaker 30 % further from the microphone),
    simulating larger tables or more spread-out seating arrangements.
    """
    if n <= 0:
        return []
    cx, cy = room_dims[0] / 2.0, room_dims[1] / 2.0
    a = (_TABLE_X / 2.0 + 0.3) * distance_scale  # extra reach to chair perimeter
    b = (_TABLE_Y / 2.0 + 0.3) * distance_scale
    positions: list[list[float]] = []
    for i in range(n):
        # Start at the short axis and walk around so 2-person meetings face each
        # other across the long axis.
        theta = math.pi / 2.0 + 2.0 * math.pi * i / n
        x = cx + a * math.cos(theta)
        y = cy + b * math.sin(theta)
        positions.append([float(x), float(y), float(_SPEAKER_HEIGHT_M)])
    return positions


def _ac_unit_position(room_dims: list[float]) -> list[float]:
    rx, ry, rz = room_dims
    ox, oy, oz = _AC_UNIT_OFFSET_M
    return [float(rx - ox), float(ry - oy), float(rz - oz)]


def _event_position(room_dims: list[float], rng: random.Random) -> list[float]:
    """Random table-adjacent position for an ambient sound event."""
    rx, ry, _rz = room_dims
    # Keep events inside the room with a half-metre margin and away from the
    # speakers' immediate table footprint.
    x = rng.uniform(0.5, rx - 0.5)
    y = rng.uniform(0.5, ry - 0.5)
    z = rng.uniform(0.5, 1.8)
    return [float(x), float(y), float(z)]


# ---------------------------------------------------------------------------
# Audio helpers (lazy numpy/soundfile import — only needed when enabled)
# ---------------------------------------------------------------------------


def _load_wav(path: Path, target_sr: int):
    """Load ``path`` as float32 mono at ``target_sr`` (linear resample)."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        ratio = target_sr / sr
        n = int(len(audio) * ratio)
        idx = np.linspace(0, len(audio) - 1, n)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
    return audio.astype("float32")


def _write_wav(path: Path, audio, sr: int) -> None:
    import soundfile as sf

    fileio.makedirs(path.parent)
    sf.write(str(path), audio, sr)


def _tile_to_length(audio, length: int, start_offset: int = 0):
    """Tile ``audio`` to ``length`` samples, optionally starting at an offset.

    ``start_offset`` applies a circular shift to ``audio`` *before* tiling so
    two callers using the same source file but different offsets produce
    distinct tiled patterns. This matters for stationary background tracks
    (background noise, AC unit) tiled across multi-minute meetings: without an
    offset, every meeting that picks the same source file gets the exact
    same waveform.
    """
    import numpy as np

    if len(audio) == 0:
        return np.zeros(length, dtype="float32")
    if start_offset:
        start_offset = int(start_offset) % len(audio)
        if start_offset:
            audio = np.concatenate([audio[start_offset:], audio[:start_offset]])
    if len(audio) >= length:
        return audio[:length].copy()
    reps = int(math.ceil(length / len(audio)))
    return np.tile(audio, reps)[:length].astype("float32")


def _rms(audio) -> float:
    import numpy as np

    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _scale_to_snr(noise, speech_rms: float, snr_db: float):
    """Scale ``noise`` so its RMS sits ``snr_db`` dB below ``speech_rms``."""

    noise_rms = _rms(noise)
    if noise_rms <= 0 or speech_rms <= 0:
        return noise * 0.0
    target = speech_rms / (10.0 ** (snr_db / 20.0))
    return (noise * (target / noise_rms)).astype("float32")


def _seeded_rng(seed: int, *parts: object) -> random.Random:
    h = hashlib.sha1(str((seed, *parts)).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------------------
# Noise asset selection
# ---------------------------------------------------------------------------


_DEFAULT_BACKGROUND_ENVIRONMENTS = ("office",)


def _background_wav_candidates(
    background_noise_dir: Path,
    environments: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """List every background WAV under the selected environment folders.

    Pass ``environments`` to scope the pool to specific subfolders of
    ``background_noise_dir`` (e.g. ``["office", "hallway"]``). When
    ``None`` we fall back to ``_DEFAULT_BACKGROUND_ENVIRONMENTS``; any listed
    environment that isn't staged on disk is silently skipped.
    """
    if not background_noise_dir or not background_noise_dir.exists():
        return []
    envs = tuple(environments) if environments else _DEFAULT_BACKGROUND_ENVIRONMENTS
    candidates: list[Path] = []
    for env in envs:
        env_dir = background_noise_dir / env
        if env_dir.is_dir():
            candidates.extend(p for p in sorted(env_dir.glob("*.wav")) if p.is_file())
    if not candidates:
        candidates = [p for p in sorted(background_noise_dir.rglob("*.wav")) if p.is_file()]
    # Stable order, unique paths (e.g. symlinks).
    uniq = {str(p.resolve()): p for p in candidates}
    return sorted(uniq.values(), key=lambda p: str(p))


def _select_background_file(
    background_noise_dir: Path,
    rng: random.Random,
    environments: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    """Pick a background WAV using ``rng`` (caller supplies seeded RNG)."""
    candidates = _background_wav_candidates(background_noise_dir, environments)
    if not candidates:
        return None
    return rng.choice(candidates)


def _select_ac_unit_file(hum_dir: Path, rng: random.Random) -> Path | None:
    if not hum_dir or not hum_dir.exists():
        return None
    # Layout: user-supplied <hum_dir>/ac_unit/*.wav files.
    wavs = sorted((hum_dir / "ac_unit").glob("*.wav"))
    return rng.choice(wavs) if wavs else None


def _event_index(event_noise_dir: Path) -> dict[str, list[Path]]:
    """Build category → list[Path] map from the event index, if present."""
    if not event_noise_dir or not event_noise_dir.exists():
        return {}
    # Layout: <root>/meta/events.csv (filename,category) + <root>/audio/*.wav.
    csv_path = event_noise_dir / "meta" / "events.csv"
    if not csv_path.exists():
        return {}
    audio_root = event_noise_dir / "audio"
    index: dict[str, list[Path]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("category", "")
            fname = row.get("filename", "")
            wav = audio_root / fname
            if wav.exists():
                index.setdefault(cat, []).append(wav)
    return index


# ---------------------------------------------------------------------------
# Wall materials.
#
# We support two modes:
#
# 1. ``randomize_wall_materials = False`` (legacy): a fixed scalar α per
#    surface that reproduces the original hand-tuned absorption used before
#    randomization was introduced.
#
# 2. ``randomize_wall_materials = True`` (default): sample a complete
#    office "material profile" per meeting using an RNG seeded by the
#    config seed and the meeting id, then sample one material per surface
#    from that profile's candidate list. Each profile is a plausible style
#    of office meeting room (modern carpeted with acoustic ceiling tile,
#    hard-floored open plan, wood-paneled executive, heavily-treated
#    conference, plus the legacy scalar profile preserved as one option).
#
# Material names are looked up in the pyroomacoustics absorption database
# (``pra.materials_data``) and yield frequency-dependent absorption curves.
# Within a single profile we keep the materials homogeneous in spirit (the
# legacy profile uses scalar α; the others use named multi-band materials)
# so that the room behaves coherently. Mixing 6- and 7-band materials
# across surfaces is supported by pyroomacoustics.
# ---------------------------------------------------------------------------

_WALL_ALPHA = {
    "east": 0.05,
    "west": 0.05,
    "north": 0.10,
    "south": 0.05,
    "floor": 0.35,
    "ceiling": 0.70,
}

_WALL_FACES = ("east", "west", "north", "south")


@dataclass(frozen=True)
class _OfficeMaterialProfile:
    """A plausible set of pyroomacoustics materials for an office room.

    ``wall_candidates`` / ``floor_candidates`` / ``ceiling_candidates`` may
    contain either ``str`` (looked up in ``pra.materials_data``) or
    ``float`` (interpreted as a scalar energy-absorption coefficient).
    A material is sampled per-surface independently from those lists.
    With probability ``glass_wall_probability`` and a non-empty
    ``glass_wall_candidates`` list, exactly one of the four walls is then
    overridden with a glass material to model an exterior window or glass
    partition — a near-universal feature of real office rooms.
    """

    name: str
    wall_candidates: tuple[str | float, ...]
    floor_candidates: tuple[str | float, ...]
    ceiling_candidates: tuple[str | float, ...]
    glass_wall_candidates: tuple[str, ...] = ()
    glass_wall_probability: float = 0.0
    weight: float = 1.0


_OFFICE_MATERIAL_PROFILES: tuple[_OfficeMaterialProfile, ...] = (
    # Legacy hand-tuned scalar α. Walls at α=0.05 are very reflective —
    # think bare unpainted gypsum or polished concrete — so the resulting
    # RT60 is noticeably long. Kept as a low-weight sampling option for
    # diversity and back-compat with prior runs, but no longer the
    # dominant profile.
    _OfficeMaterialProfile(
        name="default_legacy",
        wall_candidates=(_WALL_ALPHA["east"],),
        floor_candidates=(_WALL_ALPHA["floor"],),
        ceiling_candidates=(_WALL_ALPHA["ceiling"],),
        weight=0.5,
    ),
    # Most common modern office: gypsum/drywall walls, low-pile carpet,
    # suspended acoustic ceiling tile. Realistic mid-RT60 (~0.4-0.5 s)
    # given the updated ray-tracing tail controls.
    _OfficeMaterialProfile(
        name="modern_carpet_acoustic_ceiling",
        wall_candidates=("plasterboard", "wooden_lining"),
        floor_candidates=(
            "carpet_tufted_9.5mm",
            "carpet_thin",
            "carpet_soft_10mm",
            "carpet_6mm_closed_cell_foam",
        ),
        ceiling_candidates=(
            "ceiling_fissured_tile",
            "ceiling_perforated_gypsum_board",
        ),
        glass_wall_candidates=("glass_window", "double_glazing_10mm"),
        glass_wall_probability=0.6,
        weight=4.0,
    ),
    # Open-plan office or huddle room with hard floors and minimal
    # treatment — produces noticeably longer RT60. Kept at modest weight
    # so most meetings land in friendlier rooms while still exercising
    # the reverberant tail of the distribution.
    _OfficeMaterialProfile(
        name="hard_floor_open_plan",
        wall_candidates=("plasterboard", "rough_concrete"),
        floor_candidates=("linoleum_on_concrete", "concrete_floor"),
        ceiling_candidates=("ceiling_plasterboard", "ceiling_fissured_tile"),
        glass_wall_candidates=("glass_window", "double_glazing_10mm"),
        glass_wall_probability=0.6,
        weight=1.0,
    ),
    # Wood-paneled executive boardroom: warmer mid-band absorption,
    # carpeted floor, plain ceiling.
    _OfficeMaterialProfile(
        name="executive_wood_panel",
        wall_candidates=("wooden_lining", "wood_16mm", "plywood_thin"),
        floor_candidates=(
            "carpet_tufted_9.5mm",
            "carpet_hairy",
            "carpet_1.35_kg_m2",
        ),
        ceiling_candidates=("ceiling_plasterboard", "ceiling_fissured_tile"),
        glass_wall_candidates=("glass_window",),
        glass_wall_probability=0.3,
        weight=1.5,
    ),
    # Acoustically treated conference / call room: fabric-covered wall
    # absorbers and a high-absorption ceiling. Short RT60.
    _OfficeMaterialProfile(
        name="acoustically_treated_conference",
        wall_candidates=(
            "panel_fabric_covered_6pcf",
            "panel_fabric_covered_8pcf",
            "plasterboard",
        ),
        floor_candidates=("carpet_thin", "carpet_tufted_9.5mm"),
        ceiling_candidates=(
            "ceiling_fibre_absorber",
            "ceiling_melamine_foam",
            "ceiling_perforated_gypsum_board",
        ),
        glass_wall_candidates=("glass_window",),
        glass_wall_probability=0.2,
        weight=1.5,
    ),
)


def _make_material(pra, spec: str | float):
    """Build a ``pra.Material`` from either a database name or scalar α."""
    if isinstance(spec, str):
        return pra.Material(spec)
    return pra.Material(energy_absorption=float(spec))


def _weighted_choice(rng: random.Random, items, weights):
    """Stable weighted choice using ``rng.random()`` so seeding is honoured."""
    total = float(sum(weights))
    if total <= 0:
        return items[0]
    r = rng.random() * total
    cum = 0.0
    for item, w in zip(items, weights):
        cum += float(w)
        if r < cum:
            return item
    return items[-1]


def _build_materials(
    pra,
    rng: random.Random | None = None,
) -> tuple[dict, dict]:
    """Return ``(materials, descriptor)`` for the room's six surfaces.

    When ``rng`` is None, falls back to the legacy fixed scalar profile so
    callers that don't care about randomization (e.g. unit tests) keep
    working unchanged. Otherwise samples a profile + per-surface materials
    using ``rng``. ``descriptor`` mirrors the chosen materials by name /
    scalar so callers can persist it for verifiability.
    """
    if rng is None:
        materials = {
            face: pra.Material(energy_absorption=_WALL_ALPHA[face])
            for face in (*_WALL_FACES, "floor", "ceiling")
        }
        descriptor = {
            "profile": "default_legacy",
            "surfaces": {face: _WALL_ALPHA[face] for face in materials},
        }
        return materials, descriptor

    profile = _weighted_choice(
        rng,
        _OFFICE_MATERIAL_PROFILES,
        [p.weight for p in _OFFICE_MATERIAL_PROFILES],
    )

    surfaces: dict[str, str | float] = {}
    for face in _WALL_FACES:
        surfaces[face] = rng.choice(profile.wall_candidates)
    if (
        profile.glass_wall_candidates
        and rng.random() < profile.glass_wall_probability
    ):
        glass_face = rng.choice(_WALL_FACES)
        surfaces[glass_face] = rng.choice(profile.glass_wall_candidates)
    surfaces["floor"] = rng.choice(profile.floor_candidates)
    surfaces["ceiling"] = rng.choice(profile.ceiling_candidates)

    materials = {face: _make_material(pra, spec) for face, spec in surfaces.items()}
    descriptor = {"profile": profile.name, "surfaces": dict(surfaces)}
    return materials, descriptor


# ---------------------------------------------------------------------------
# ShoeBox construction
# ---------------------------------------------------------------------------


def _build_shoebox(
    pra,
    dims: list[float],
    sr: int,
    materials: dict,
    room_cfg: RoomSimulationSettings,
    time_thres_override: float | None = None,
):
    """Construct a ``pra.ShoeBox`` and apply ray-tracing tail controls.

    The pyroomacoustics default ray-tracer accumulates reflected energy out
    to ``time_thres=10.0 s`` with ``energy_thres=1e-7``, which produces an
    unrealistically long late tail for small rooms and is the dominant
    cause of the "echo-y" character in the unmodified simulation. When
    ``room_cfg.ray_tracing`` is True we call ``set_ray_tracing(...)`` with
    the conference-room defaults from :class:`RoomSimulationSettings`.

    ``time_thres_override`` replaces ``room_cfg.ray_tracing_time_thres_s``
    when provided, allowing per-meeting sampling without mutating the config.
    """
    room = pra.ShoeBox(
        dims,
        fs=sr,
        materials=materials,
        max_order=int(room_cfg.max_order),
        ray_tracing=bool(room_cfg.ray_tracing),
        air_absorption=bool(room_cfg.air_absorption),
    )
    if bool(room_cfg.ray_tracing):
        time_thres = (
            time_thres_override
            if time_thres_override is not None
            else float(room_cfg.ray_tracing_time_thres_s)
        )
        room.set_ray_tracing(
            time_thres=time_thres,
            energy_thres=float(room_cfg.ray_tracing_energy_thres),
        )
    return room


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def simulate_room(
    in_room_speaker_wavs: dict[str, Path],
    timing_data: dict,
    join_spec: MeetingJoinSpec,
    noise_paths: RoomNoisePaths,
    room_cfg: RoomSimulationSettings,
    out_dir: Path,
    seed: int,
    meeting_id: str = "",
) -> RoomSimulationResult:
    """Apply room acoustics + background noise to in-room speaker tracks.

    Inputs are per-speaker WAVs from step 9 (one per in-room speaker). The
    function convolves each independently with the room RIR, sums the results
    into a single microphone signal, layers noise, and writes both per-speaker
    room-convolved WAVs and the combined room recording.
    """
    import numpy as np
    import pyroomacoustics as pra

    sr = int(room_cfg.sample_rate)
    fileio.makedirs(out_dir)

    # ------------------------------------------------------------------
    # 0. Sample per-meeting acoustic difficulty parameters.
    # ------------------------------------------------------------------
    diff_rng = _seeded_rng(seed, "difficulty", meeting_id)

    def _uniform(lo: float, hi: float) -> float:
        return lo if lo >= hi else diff_rng.uniform(lo, hi)

    tt_lo, tt_hi = room_cfg.ray_tracing_time_thres_range
    time_thres_sampled = _uniform(tt_lo, tt_hi)

    rs_lo, rs_hi = room_cfg.room_size_scale_range
    room_scale = _uniform(rs_lo, rs_hi)

    ac_lo, ac_hi = room_cfg.ac_boost_db_range
    ac_boost_db = _uniform(ac_lo, ac_hi)

    logger.info(
        "Meeting %s difficulty sample: time_thres=%.3f s  room_scale=%.3f  "
        "ac_boost_db=%.1f dB",
        meeting_id or "<unknown>",
        time_thres_sampled,
        room_scale,
        ac_boost_db,
    )

    # ------------------------------------------------------------------
    # 1. Load in-room speaker signals (in deterministic order).
    # ------------------------------------------------------------------
    speaker_keys = sorted(in_room_speaker_wavs.keys())
    if not speaker_keys:
        raise ValueError("simulate_room called with no in-room speakers")

    signals = [_load_wav(in_room_speaker_wavs[k], sr) for k in speaker_keys]
    max_len = max(len(s) for s in signals)
    signals = [
        np.pad(s, (0, max_len - len(s))).astype("float32") if len(s) < max_len else s
        for s in signals
    ]

    # ------------------------------------------------------------------
    # 2. Build the conference room and place sources / microphone.
    # ------------------------------------------------------------------
    materials_rng = (
        _seeded_rng(seed, "wall_materials", meeting_id)
        if room_cfg.randomize_wall_materials
        else None
    )
    materials, materials_descriptor = _build_materials(pra, materials_rng)
    logger.info(
        "Room materials for meeting %s: profile=%s surfaces=%s",
        meeting_id or "<unknown>",
        materials_descriptor["profile"],
        materials_descriptor["surfaces"],
    )
    base_dims = list(room_cfg.room_dims)
    # Scale X/Y footprint; keep ceiling height fixed so the room stays
    # acoustically plausible (floor-to-ceiling ~2.7 m regardless of scale).
    dims = [base_dims[0] * room_scale, base_dims[1] * room_scale, base_dims[2]]
    room = _build_shoebox(pra, dims, sr, materials, room_cfg,
                          time_thres_override=time_thres_sampled)

    mic_pos = np.array([[dims[0] / 2.0], [dims[1] / 2.0], [_MIC_HEIGHT_M]])
    room.add_microphone_array(mic_pos)

    positions = conference_table_positions(len(speaker_keys), dims,
                                           distance_scale=room_scale)
    for pos, signal in zip(positions, signals):
        room.add_source(pos, signal=signal)

    room.simulate()
    mic_signals = room.mic_array.signals  # shape (1, T)
    combined_speech = mic_signals[0].astype("float32")

    rt60 = 0.0
    try:
        # pra >= 0.7 exposes measure_rt60 per source; fall back gracefully.
        rt60 = float(np.median(room.measure_rt60()))
    except Exception:  # pragma: no cover — purely diagnostic
        pass

    # ------------------------------------------------------------------
    # 3. Per-speaker room-convolved tracks (one source at a time).
    # ------------------------------------------------------------------
    speaker_room_paths: dict[str, Path] = {}
    for key, signal, pos in zip(speaker_keys, signals, positions):
        solo = _build_shoebox(pra, dims, sr, materials, room_cfg,
                              time_thres_override=time_thres_sampled)
        solo.add_microphone_array(mic_pos)
        solo.add_source(pos, signal=signal)
        solo.simulate()
        solo_signal = solo.mic_array.signals[0].astype("float32")
        sp_path = out_dir / _speaker_room_filename(key)
        _write_wav(sp_path, solo_signal, sr)
        speaker_room_paths[key] = sp_path

    # ------------------------------------------------------------------
    # 4. Noise layers (added in the mic domain).
    # ------------------------------------------------------------------
    speech_rms = max(_rms(combined_speech), 1e-9)
    target_len = len(combined_speech)
    noise_track = np.zeros(target_len, dtype="float32")
    noise_config: dict = {
        "rt60_measured": rt60,
        "wall_materials": materials_descriptor,
        "background_file": None,
        "ac_unit_file": None,
        "sound_events": [],
        "white_noise_db": room_cfg.white_noise_db,
        "difficulty": {
            "time_thres_s": time_thres_sampled,
            "room_scale": room_scale,
            "ac_boost_db": ac_boost_db,
        },
    }

    # 4a. Background noise.
    background_rng = _seeded_rng(seed, "background", meeting_id)
    background_path = (
        _select_background_file(
            noise_paths.background_noise_dir, background_rng, room_cfg.background_environments,
        )
        if noise_paths.background_noise_dir
        else None
    )
    if background_path is not None:
        background_audio = _load_wav(background_path, sr)
        # Circular-shift the source so meetings that land on the same background
        # channel still produce distinct tiled patterns. Use the same RNG
        # that chose the file so the offset is deterministic per
        # (seed, meeting_id) as well.
        background_offset = background_rng.randint(0, max(len(background_audio) - 1, 0))
        background_audio = _tile_to_length(background_audio, target_len, start_offset=background_offset)
        background_scaled = _scale_to_snr(background_audio, speech_rms, room_cfg.background_snr_db)
        noise_track = noise_track + background_scaled
        noise_config["background_file"] = str(background_path)
        noise_config["background_environment"] = background_path.parent.name
        noise_config["background_offset_samples"] = int(background_offset)
    else:
        logger.info("No background noise file available; skipping background layer.")

    # 4b. AC unit (RIR-convolved from a corner of the room).
    ac_rng = _seeded_rng(seed, "ac_unit", meeting_id)
    ac_path = _select_ac_unit_file(noise_paths.hum_noise_dir, ac_rng) if noise_paths.hum_noise_dir else None
    if ac_path is not None:
        ac_audio = _load_wav(ac_path, sr)
        # Same circular-shift trick as the background: stops repeat AC clips from
        # producing byte-identical hum across meetings.
        ac_offset = ac_rng.randint(0, max(len(ac_audio) - 1, 0))
        ac_audio = _tile_to_length(ac_audio, target_len, start_offset=ac_offset)
        ac_room = _build_shoebox(pra, dims, sr, materials, room_cfg,
                                 time_thres_override=time_thres_sampled)
        ac_room.add_microphone_array(mic_pos)
        ac_room.add_source(_ac_unit_position(dims), signal=ac_audio)
        ac_room.simulate()
        ac_mic = ac_room.mic_array.signals[0].astype("float32")
        # Trim/pad to match speech length.
        if len(ac_mic) >= target_len:
            ac_mic = ac_mic[:target_len]
        else:
            ac_mic = np.pad(ac_mic, (0, target_len - len(ac_mic)))
        # ac_boost_db reduces the below-background gap, making the AC louder.
        ac_target_db = room_cfg.background_snr_db + room_cfg.ac_unit_db_below_background - ac_boost_db
        ac_scaled = _scale_to_snr(ac_mic, speech_rms, ac_target_db)
        noise_track = noise_track + ac_scaled
        noise_config["ac_unit_file"] = str(ac_path)
    else:
        logger.info("No AC-unit file available; skipping AC hum.")

    # 4c. Short sound events.
    event_index = _event_index(noise_paths.event_noise_dir) if noise_paths.event_noise_dir else {}
    if event_index and room_cfg.events_per_minute > 0:
        duration_min = target_len / float(sr) / 60.0
        n_events = int(round(duration_min * room_cfg.events_per_minute))
        if n_events > 0:
            event_rng = _seeded_rng(seed, "events", meeting_id)
            categories = [
                c for c in room_cfg.event_categories if c in event_index
            ]
            if not categories:
                logger.info(
                    "No matching event categories found in %s", noise_paths.event_noise_dir,
                )
            else:
                esc_target_db = room_cfg.background_snr_db + room_cfg.event_snr_db
                for idx in range(n_events):
                    cat = event_rng.choice(categories)
                    wav_path = event_rng.choice(event_index[cat])
                    event_audio = _load_wav(wav_path, sr)
                    if len(event_audio) == 0:
                        continue
                    pos = _event_position(dims, event_rng)
                    ev_room = _build_shoebox(pra, dims, sr, materials, room_cfg,
                                             time_thres_override=time_thres_sampled)
                    ev_room.add_microphone_array(mic_pos)
                    ev_room.add_source(pos, signal=event_audio)
                    ev_room.simulate()
                    ev_mic = ev_room.mic_array.signals[0].astype("float32")

                    start_sample = event_rng.randint(0, max(0, target_len - 1))
                    end_sample = min(start_sample + len(ev_mic), target_len)
                    ev_clip = ev_mic[: end_sample - start_sample]
                    ev_scaled = _scale_to_snr(ev_clip, speech_rms, esc_target_db)
                    noise_track[start_sample:end_sample] += ev_scaled
                    noise_config["sound_events"].append(
                        {
                            "category": cat,
                            "path": str(wav_path),
                            "start_sample": start_sample,
                            "position": pos,
                        }
                    )

    # 4d. White-noise floor (constant dBFS irrespective of speech level).
    white_rng = _seeded_rng(seed, "white_noise", meeting_id)
    np_rng = np.random.default_rng(white_rng.randint(0, 2**31 - 1))
    white = np_rng.standard_normal(target_len).astype("float32")
    white_rms_target = 10.0 ** (room_cfg.white_noise_db / 20.0)
    white_rms = max(_rms(white), 1e-9)
    white = (white * (white_rms_target / white_rms)).astype("float32")
    noise_track = noise_track + white

    # ------------------------------------------------------------------
    # 5. Final mix and write.
    # ------------------------------------------------------------------
    mix = combined_speech + noise_track
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.99:
        mix = (mix * (0.99 / peak)).astype("float32")

    combined_path = out_dir / "combined_room.wav"
    _write_wav(combined_path, mix, sr)

    return RoomSimulationResult(
        combined_room_path=combined_path,
        speaker_room_paths=speaker_room_paths,
        rt60_measured=rt60,
        noise_config=noise_config,
    )


def _speaker_room_filename(speaker_key: str) -> str:
    """Map a speaker key to a stable on-disk WAV name."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in speaker_key)
    return f"speaker_{safe}_room.wav"


def mix_room_with_remote_tracks(
    combined_room_path: Path,
    remote_speaker_wavs: list[Path],
    out_path: Path,
    sample_rate: int,
) -> Path:
    """Sum the room mic recording with dry remote/phone speaker tracks.

    The room recording from ``simulate_room`` contains in-room speakers
    convolved with the room RIR plus background noise. Step 9's dry
    per-speaker WAVs are full meeting-length tracks with silence outside
    the speaker's turns, so they can be summed directly onto the room mic
    to produce a single "room + dial-in" recording per meeting.

    All inputs are resampled to ``sample_rate`` and zero-padded to the
    longest signal. The mixed waveform is peak-normalised to 0.99 if
    needed (mirroring ``simulate_room``). When ``remote_speaker_wavs``
    is empty the room file is byte-faithfully copied so callers can rely
    on ``out_path`` always existing after a successful call.
    """
    import shutil

    import numpy as np

    if not combined_room_path.exists():
        raise FileNotFoundError(
            f"combined_room.wav not found at {combined_room_path}",
        )

    fileio.makedirs(out_path.parent)

    if not remote_speaker_wavs:
        shutil.copyfile(str(combined_room_path), str(out_path))
        return out_path

    room_audio = _load_wav(combined_room_path, sample_rate)
    remote_signals = [_load_wav(p, sample_rate) for p in remote_speaker_wavs]

    max_len = max(len(room_audio), *(len(s) for s in remote_signals))

    def _pad(sig):
        if len(sig) >= max_len:
            return sig[:max_len].astype("float32")
        return np.pad(sig, (0, max_len - len(sig))).astype("float32")

    mix = _pad(room_audio)
    for sig in remote_signals:
        mix = mix + _pad(sig)
    mix = mix.astype("float32")

    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.99:
        mix = (mix * (0.99 / peak)).astype("float32")

    _write_wav(out_path, mix, sample_rate)
    return out_path


# ---------------------------------------------------------------------------
# Helpers for callers (step orchestration)
# ---------------------------------------------------------------------------


def in_room_speaker_ids(join_spec: MeetingJoinSpec) -> set[str]:
    return {
        pid
        for pid, info in join_spec.participant_modes.items()
        if info.mode == ParticipantJoinMode.IN_ROOM
    }


def remote_speaker_ids(join_spec: MeetingJoinSpec) -> set[str]:
    """Participant IDs joining remotely (``REMOTE`` or ``PHONE``).

    Step 10 uses this to identify which step-9 dry per-speaker tracks should
    be summed onto the room mic recording for the final per-meeting mix.
    """
    return {
        pid
        for pid, info in join_spec.participant_modes.items()
        if info.mode in (ParticipantJoinMode.REMOTE, ParticipantJoinMode.PHONE)
    }


def resolve_noise_paths(
    cfg: RoomSimulationSettings,
    output_root: str | Path,
) -> RoomNoisePaths:
    """Resolve relative noise-directory paths against ``output_root``.

    A configured directory that does not exist is a config bug, not a runtime
    skip — raise rather than silently dropping that noise layer.
    """
    base = Path(output_root)

    def _resolve(rel: str, label: str) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        if not p.is_absolute():
            p = base / rel
        if not p.exists():
            raise FileNotFoundError(
                f"Room simulation {label} not found: {p}. Supply local noise "
                f"files in the layout described in README's 'Local noise files' "
                f"section, update the path, or set it to an empty string to omit this layer.",
            )
        return p

    return RoomNoisePaths(
        background_noise_dir=_resolve(cfg.background_noise_dir, "background_noise_dir"),
        hum_noise_dir=_resolve(cfg.hum_noise_dir, "hum_noise_dir"),
        event_noise_dir=_resolve(cfg.event_noise_dir, "event_noise_dir"),
    )
