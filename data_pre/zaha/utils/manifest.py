"""manifest.json schema + D-21 sanity checks + D-22 hard-failure policy.

RESEARCH §H.2 schema writer for Plan 04's orchestrator. The per-sample /
per-chunk metadata is produced by ``build_zaha_chunks.py`` and serialized via
``write_manifest``. Downstream tooling (the verifier, training configs) reads
``manifest.json`` as the authoritative contract for the chunked dataset.

Dataclass layout mirrors RESEARCH §H.2 literally: ``Manifest`` → list of
``SampleEntry`` → list of ``ChunkEntry``. All dict fields use ``str`` keys so
``json.dump`` round-trips losslessly without the orchestrator having to
stringify on every path.

D-21 sanity gates
-----------------
``run_sanity_checks`` returns a list of human-readable error strings. Any
string that starts with ``"HARD FAIL"`` is a D-22 hard-failure trigger and
the orchestrator must ``sys.exit`` non-zero. Warnings (drift between 5 pp and
15 pp) are logged but do not block the build.

Gates implemented
~~~~~~~~~~~~~~~~~
(a) per-class histogram drift — compares the raw-class histogram (0..16,
    VOID included) against the final histogram (0..15, post-remap) after
    shifting the raw one-index-down to the remapped space. Drift > 15 pp
    on any class is a HARD FAIL.
(b) chunk budget — any chunk with ``point_count > 1_000_000`` (D-07
    supersession 2026-04-12) is a HARD FAIL.
(c) readme coverage — the manifest must contain exactly the samples the
    caller expected (via the ZAHA readme.txt split assignments). Missing
    or extra samples are HARD FAILs.

Import order
------------
This module imports ``json``, ``subprocess``, ``platform``, ``getpass``,
``datetime`` — all stdlib. It does NOT import numpy, pandas, scipy, or
open3d, so it is safe to import before or after any of those.
"""
from __future__ import annotations

import getpass
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


#: Schema version for manifest.json. Bump when adding/removing top-level keys.
SCHEMA_VERSION: int = 1

#: Pipeline version recorded into manifest.json alongside the git commit hash.
PIPELINE_VERSION: str = "1.0.0"

#: D-07 per-chunk point budget (supersession 2026-04-12 — Plan 01-04 Task 3):
#: the original 600k cap sat on top of grid=0.02 + fixed-4m-tile; under
#: grid=0.04 + adaptive continuous sizing (TARGET_PTS=400k) typical chunks
#: land in 200k-700k and the 1M cap is the genuine overflow bound.
_BUDGET_PER_CHUNK: int = 1_000_000

#: Per-class histogram drift thresholds (fractions, not percentages).
_DRIFT_WARN: float = 0.05  # 5 pp warning
_DRIFT_FAIL: float = 0.15  # 15 pp hard fail


# ---------------------------------------------------------------------------
# Dataclasses — RESEARCH §H.2 schema
# ---------------------------------------------------------------------------


@dataclass
class ChunkEntry:
    """Per-chunk manifest entry — one row per ``__c<idx>`` directory."""

    chunk_idx: int
    dir_name: str
    x_tile: int
    y_tile: int
    bbox_min: list[float]
    bbox_max: list[float]
    point_count: int
    class_histogram: dict[str, int]  # str keys for json round-trip
    coord_sha256: str
    segment_sha256: str
    normal_sha256: str


@dataclass
class SampleEntry:
    """Per-sample manifest entry — one row per ``<sample>.pcd``."""

    sample: str
    split: str
    source_pcd: str
    source_pcd_sha256: str
    raw_point_count: int
    post_downsample_voxel_count: int
    post_void_drop_voxel_count: int
    post_denoise_point_count: int
    bbox_min: list[float]
    bbox_max: list[float]
    chunks: list[ChunkEntry]
    elapsed_s: float
    peak_rss_mb: float
    class_histogram_raw: dict[str, int]  # raw 0..16 (VOID included)
    class_histogram_final: dict[str, int]  # remapped 0..15 (no VOID)


@dataclass
class Manifest:
    """Top-level manifest payload serialized to ``manifest.json``."""

    schema_version: int
    pipeline_version: str
    commit_hash: str
    ran_at: str  # ISO-8601 UTC
    ran_by: str
    host: str
    grid_size: float
    void_drop_rule: str
    denoising: dict
    normal_estimation: dict
    chunking: dict
    dataset_stats: dict
    samples: list[SampleEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def get_commit_hash() -> str:
    """Return ``git rev-parse HEAD`` with a ``-dirty`` suffix if uncommitted.

    Returns ``"unknown"`` if the caller is not inside a git repo or ``git``
    is not on ``PATH`` — both cases happen in CI sandboxes that strip the
    ``.git`` directory.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        return sha + ("-dirty" if dirty != 0 else "")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def build_manifest_shell(
    denoising: dict,
    normal_estimation: dict,
    chunking: dict,
) -> Manifest:
    """Assemble an empty-samples ``Manifest`` with provenance filled in."""
    return Manifest(
        schema_version=SCHEMA_VERSION,
        pipeline_version=PIPELINE_VERSION,
        commit_hash=get_commit_hash(),
        ran_at=datetime.now(timezone.utc).isoformat(),
        ran_by=getpass.getuser(),
        host=platform.node(),
        grid_size=0.04,
        void_drop_rule="winner_eq_0_drops_voxel",
        denoising=denoising,
        normal_estimation=normal_estimation,
        chunking=chunking,
        dataset_stats={},
    )


# ---------------------------------------------------------------------------
# Dataset stats aggregation
# ---------------------------------------------------------------------------


def aggregate_dataset_stats(samples: list[SampleEntry]) -> dict:
    """Roll up per-sample counters into a single ``dataset_stats`` dict."""
    raw: dict[str, int] = {str(i): 0 for i in range(17)}  # raw includes VOID
    final: dict[str, int] = {str(i): 0 for i in range(16)}
    for s in samples:
        for k, v in s.class_histogram_raw.items():
            raw[str(k)] = raw.get(str(k), 0) + int(v)
        for k, v in s.class_histogram_final.items():
            final[str(k)] = final.get(str(k), 0) + int(v)
    total_raw = sum(int(s.raw_point_count) for s in samples)
    total_down = sum(int(s.post_downsample_voxel_count) for s in samples)
    total_void = sum(int(s.post_void_drop_voxel_count) for s in samples)
    total_denoise = sum(int(s.post_denoise_point_count) for s in samples)
    total_chunks = sum(len(s.chunks) for s in samples)
    return {
        "raw_total_points": int(total_raw),
        "post_downsample_total_points": int(total_down),
        "post_void_drop_total_points": int(total_void),
        "post_denoise_total_points": int(total_denoise),
        "total_chunks": int(total_chunks),
        "class_histogram_raw": raw,
        "class_histogram_final": final,
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_manifest(out_dir: Path, manifest: Manifest) -> Path:
    """Serialize ``manifest`` to ``<out_dir>/manifest.json`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "manifest.json"
    payload = asdict(manifest)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return out


# ---------------------------------------------------------------------------
# D-21 sanity checks
# ---------------------------------------------------------------------------


def run_sanity_checks(
    manifest: Manifest,
    expected_samples: set[str],
) -> list[str]:
    """D-21 gates. Returns a list of human-readable error strings.

    Any string starting with ``"HARD FAIL"`` must trigger D-22 hard-failure
    (non-zero exit) in the orchestrator; other strings are warnings that
    should be logged but do not block the build.

    Parameters
    ----------
    manifest : Manifest
        Fully-populated manifest (must have ``dataset_stats`` computed).
    expected_samples : set[str]
        The set of sample basenames the caller expected to see — normally
        derived from ``readme.txt`` under ``ZAHA_pcd/``.

    Returns
    -------
    list[str]
        Empty list on a clean pass. Otherwise a list of error strings, each
        prefixed with either ``"HARD FAIL"`` or ``"WARN"``.
    """
    errors: list[str] = []

    # (a) per-class histogram drift ----------------------------------------
    raw = manifest.dataset_stats.get("class_histogram_raw", {})
    final = manifest.dataset_stats.get("class_histogram_final", {})
    # Shift raw (0..16 incl VOID) one step down into the remapped 0..15
    # space by dropping the VOID key. raw key '0' = VOID, raw key 'k' for
    # k>0 corresponds to remapped class (k - 1).
    raw_remapped: dict[int, int] = {}
    for k, v in raw.items():
        kk = int(k)
        if kk > 0:
            raw_remapped[kk - 1] = int(v)
    total_raw = sum(raw_remapped.values())
    total_final = sum(int(v) for v in final.values())
    if total_raw > 0 and total_final > 0:
        for k in range(16):
            p_raw = raw_remapped.get(k, 0) / total_raw
            p_final = int(final.get(str(k), 0)) / total_final
            drift = abs(p_final - p_raw)
            if drift > _DRIFT_FAIL:
                errors.append(
                    f"HARD FAIL class {k} drift "
                    f"{drift*100:.1f}pp > {_DRIFT_FAIL*100:.0f}pp"
                )
            elif drift > _DRIFT_WARN:
                errors.append(
                    f"WARN class {k} drift "
                    f"{drift*100:.1f}pp > {_DRIFT_WARN*100:.0f}pp"
                )
    else:
        errors.append(
            "WARN: empty histograms — sanity drift check skipped"
        )

    # (b) chunk budget enforcement (D-07) ---------------------------------
    for sample in manifest.samples:
        for chunk in sample.chunks:
            if int(chunk.point_count) > _BUDGET_PER_CHUNK:
                errors.append(
                    f"HARD FAIL {chunk.dir_name} over budget: "
                    f"{int(chunk.point_count)} > {_BUDGET_PER_CHUNK}"
                )

    # (c) readme coverage -------------------------------------------------
    actual = {s.sample for s in manifest.samples}
    missing = expected_samples - actual
    if missing:
        errors.append(
            f"HARD FAIL missing samples: {sorted(missing)}"
        )
    extra = actual - expected_samples
    if extra:
        errors.append(
            f"HARD FAIL unexpected samples: {sorted(extra)}"
        )

    return errors
