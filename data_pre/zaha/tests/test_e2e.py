"""End-to-end tests for the ZAHA orchestrator — DS-ZAHA-P1-07.

GREEN in Plan 04 Task 2 — ``build_zaha_chunks.py`` runs on the smallest
sample, emits a valid manifest, passes sanity gates, and is bitwise
deterministic across reruns. ``test_golden`` stays RED until Task 3 writes
the golden file; ``test_hard_failure_on_missing_sample`` is an optional
follow-up.

Target impl: ``data_pre.zaha.scripts.build_zaha_chunks`` (Plan 04).
Behaviour references: CONTEXT.md D-20 (manifest schema), D-21 (sanity gates),
D-22 (hard-failure policy), RESEARCH.md §H (orchestration), VALIDATION.md
rows ``P01-e2e-01..04`` + ``P01-e2e-golden``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Resolve the repo root relative to this test file:
#   parents[0] = tests/
#   parents[1] = zaha/
#   parents[2] = data_pre/
#   parents[3] = repo root (sbf-net/)
REPO_ROOT = Path(__file__).resolve().parents[3]
SMALLEST = "DEBY_LOD2_4907179"
SCRIPT = REPO_ROOT / "data_pre" / "zaha" / "scripts" / "build_zaha_chunks.py"
ZAHA_PCD = Path("/home/mty0201/data/ZAHA_pcd")
GOLDEN_PATH = (
    REPO_ROOT
    / "data_pre"
    / "zaha"
    / "tests"
    / "golden"
    / "DEBY_LOD2_4907179__goldens.json"
)


def _skip_if_no_data() -> None:
    if not ZAHA_PCD.exists():
        pytest.skip(f"{ZAHA_PCD} not mounted — e2e tests require real data")
    if not SCRIPT.exists():
        pytest.skip(f"{SCRIPT} not present — Plan 04 Task 2 not run yet")


def _run_build(out_dir: Path, extra: list[str] | None = None) -> None:
    """Invoke the orchestrator and raise if it fails."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(ZAHA_PCD),
        "--output",
        str(out_dir),
        "--samples",
        SMALLEST,
        "--workers",
        "1",
        "--force",
    ]
    if extra:
        cmd.extend(extra)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"build failed (exit {proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


@pytest.fixture(scope="module")
def smallest_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the orchestrator once per module on ``DEBY_LOD2_4907179``."""
    _skip_if_no_data()
    out = tmp_path_factory.mktemp("zaha_e2e") / "chunked"
    _run_build(out)
    return out


def test_smallest_sample(smallest_run: Path) -> None:
    """P01-e2e-01 — full pipeline runs on the smallest sample without error."""
    manifest_path = smallest_run / "manifest.json"
    assert manifest_path.exists(), "manifest.json not emitted"
    chunk_dirs = sorted((smallest_run / "training").glob(f"{SMALLEST}__c*"))
    assert len(chunk_dirs) > 0, "no chunks written for smallest sample"
    for cd in chunk_dirs:
        assert (cd / "coord.npy").exists(), f"{cd} missing coord.npy"
        assert (cd / "segment.npy").exists(), f"{cd} missing segment.npy"
        assert (cd / "normal.npy").exists(), f"{cd} missing normal.npy"


def test_manifest_schema(smallest_run: Path) -> None:
    """P01-e2e-02 — ``manifest.json`` has required top-level keys (D-20)."""
    m = json.loads((smallest_run / "manifest.json").read_text())
    assert m["schema_version"] == 1
    for key in (
        "pipeline_version",
        "commit_hash",
        "grid_size",
        "denoising",
        "normal_estimation",
        "chunking",
        "dataset_stats",
        "samples",
    ):
        assert key in m, f"manifest missing required top-level key {key!r}"
    assert m["grid_size"] == 0.02
    assert len(m["samples"]) == 1
    s = m["samples"][0]
    assert s["sample"] == SMALLEST
    assert s["split"] == "training"
    assert len(s["chunks"]) >= 1
    for key in (
        "chunk_idx",
        "dir_name",
        "point_count",
        "coord_sha256",
        "segment_sha256",
        "normal_sha256",
    ):
        assert key in s["chunks"][0], f"chunk missing required field {key!r}"


def test_sanity_gates(smallest_run: Path) -> None:
    """P01-e2e-03 — D-21 sanity gates pass on a clean run and fail on tamper."""
    from data_pre.zaha.utils.manifest import (
        ChunkEntry,
        Manifest,
        SampleEntry,
        run_sanity_checks,
    )

    m_raw = json.loads((smallest_run / "manifest.json").read_text())
    sample_entries: list[SampleEntry] = []
    for sd in m_raw["samples"]:
        chunks = [ChunkEntry(**cd) for cd in sd["chunks"]]
        sample_entries.append(SampleEntry(**{**sd, "chunks": chunks}))
    manifest = Manifest(
        schema_version=m_raw["schema_version"],
        pipeline_version=m_raw["pipeline_version"],
        commit_hash=m_raw["commit_hash"],
        ran_at=m_raw["ran_at"],
        ran_by=m_raw["ran_by"],
        host=m_raw["host"],
        grid_size=m_raw["grid_size"],
        void_drop_rule=m_raw["void_drop_rule"],
        denoising=m_raw["denoising"],
        normal_estimation=m_raw["normal_estimation"],
        chunking=m_raw["chunking"],
        dataset_stats=m_raw["dataset_stats"],
        samples=sample_entries,
    )
    errors = run_sanity_checks(manifest, {SMALLEST})
    hard = [e for e in errors if "HARD FAIL" in e]
    assert not hard, f"unexpected hard failures on smallest sample: {hard}"
    # Inject a budget violation and confirm the gate fires.
    manifest.samples[0].chunks[0].point_count = 999_999_999
    errors2 = run_sanity_checks(manifest, {SMALLEST})
    hard2 = [e for e in errors2 if "HARD FAIL" in e]
    assert hard2, "expected HARD FAIL after injecting budget violation"


def test_determinism(tmp_path: Path) -> None:
    """P01-e2e-04 — two pipeline runs on the same sample are bitwise equal."""
    _skip_if_no_data()
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    for out in (out1, out2):
        _run_build(out)
    m1 = json.loads((out1 / "manifest.json").read_text())["samples"][0]["chunks"]
    m2 = json.loads((out2 / "manifest.json").read_text())["samples"][0]["chunks"]
    assert len(m1) == len(m2), f"chunk count drift: {len(m1)} vs {len(m2)}"
    for c1, c2 in zip(m1, m2):
        assert c1["dir_name"] == c2["dir_name"]
        for h in ("coord_sha256", "segment_sha256", "normal_sha256"):
            assert c1[h] == c2[h], (
                f"{c1['dir_name']} {h}: {c1[h][:16]} vs {c2[h][:16]}"
            )


def test_golden(smallest_run: Path) -> None:
    """P01-e2e-golden — outputs match frozen SHA256 hashes for regression.

    Loads ``data_pre/zaha/tests/golden/DEBY_LOD2_4907179__goldens.json``,
    compares every chunk's ``coord_sha256 / segment_sha256 / normal_sha256``
    to the frozen values. Skipped until the golden file exists (generated by
    Plan 04 Task 3).
    """
    if not GOLDEN_PATH.exists():
        pytest.skip("golden file not generated yet — run Plan 04 Task 3a")
    golden = json.loads(GOLDEN_PATH.read_text())
    m = json.loads((smallest_run / "manifest.json").read_text())
    s = next(x for x in m["samples"] if x["sample"] == SMALLEST)
    current = {c["dir_name"]: c for c in s["chunks"]}
    for dir_name, g in golden["chunks"].items():
        assert dir_name in current, f"chunk {dir_name} missing from current run"
        c = current[dir_name]
        assert c["coord_sha256"] == g["coord_sha256"], (
            f"{dir_name} coord hash drift"
        )
        assert c["segment_sha256"] == g["segment_sha256"], (
            f"{dir_name} segment hash drift"
        )
        assert c["normal_sha256"] == g["normal_sha256"], (
            f"{dir_name} normal hash drift"
        )
        assert c["point_count"] == g["point_count"]


def test_hard_failure_on_missing_sample() -> None:
    """P01-e2e-05 — orchestrator hard-fails if a requested sample is missing (D-22).

    Requesting ``--samples DEBY_LOD2_9999999`` (which does not exist on disk)
    must exit with a non-zero status. CONTEXT.md D-22 hard-failure policy.
    """
    _skip_if_no_data()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(ZAHA_PCD),
            "--output",
            str(Path(td) / "chunked"),
            "--samples",
            "DEBY_LOD2_9999999",
            "--workers",
            "1",
            "--force",
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode != 0, (
            "expected non-zero exit for missing sample, got 0"
        )
