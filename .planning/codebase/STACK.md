# Technology Stack

**Analysis Date:** 2026-04-01

## Languages

**Primary:**
- Python (version not pinned in this repo) - all executable code lives in `project/`, `scripts/`, `configs/`, and `data_pre/`.

**Secondary:**
- Markdown - operator-facing documentation in `README.md`, `install.md`, `train.md`, and `docs/data_format.md`.
- JSON - derived workflow artifacts under `reports/log_summaries/` and a minimal tooling manifest in `.claude/package.json`.

## Runtime

**Environment:**
- Python CLI runtime in a Conda environment named `ptv3`, documented in `README.md` and `install.md`.
- PyTorch chooses `cuda` when available and falls back to `cpu` in `project/trainer/trainer.py`.
- Full training is GPU-oriented; `train.md` only frames CPU fallback as a smoke/debug path.

**Package Manager:**
- `requirements.txt` at `requirements.txt` is the only repo-level dependency manifest and only lists project-local additions.
- Conda is the documented environment manager in `install.md`, but no `environment.yml` is provided here.
- Lockfile: missing. No `pyproject.toml`, `setup.py`, or `setup.cfg` detected.

## Frameworks

**Core:**
- PyTorch - tensor runtime, modules, optimizers, AMP, and dataloaders across `project/models/*.py`, `project/losses/*.py`, and `project/trainer/trainer.py`.
- Pointcept / PTv3 (external upstream dependency, version not pinned in this repo) - provides the backbone, registries, transforms, point structure utilities, collate helpers, and Lovasz loss used by `project/models/semantic_boundary_model.py`, `project/models/semantic_model.py`, `project/datasets/bf.py`, `project/trainer/trainer.py`, and `project/losses/*.py`.
- NumPy - dataset IO and numeric preprocessing in `project/datasets/bf.py`, `scripts/check_data/*.py`, and `data_pre/bf_edge_v3/**/*.py`.

**Testing:**
- No formal test framework is configured.
- Runtime verification is done with custom smoke/debug scripts in `scripts/check_data/` and `scripts/train/check_train_step.py`.

**Build/Dev:**
- Python `runpy` is the config composition mechanism in `scripts/train/train.py` and `configs/semantic_boundary/*.py`.
- Python stdlib CLI tooling powers checkpoint-summary and context-generation scripts in `scripts/agent/*.py`.
- SciPy and scikit-learn are used only in the preprocessing pipeline under `data_pre/bf_edge_v3/core/`.

## Key Dependencies

**Critical:**
- `torch` - required for model definition, optimization, AMP, and checkpointed training in `project/trainer/trainer.py`.
- `pointcept` - required upstream codebase; `scripts/train/train.py` inserts it into `sys.path`, and without it the model/dataset registries and PT-v3 backbone are unavailable.
- `numpy` - required for `.npy` / `.npz` dataset loading and preprocessing in `project/datasets/bf.py` and `data_pre/bf_edge_v3/utils/stage_io.py`.

**Infrastructure:**
- `scipy` - `cKDTree` neighbor search in `data_pre/bf_edge_v3/core/boundary_centers_core.py` and `data_pre/bf_edge_v3/core/local_clusters_core.py`.
- `scikit-learn` - `DBSCAN` clustering in `data_pre/bf_edge_v3/core/local_clusters_core.py`.
- Python stdlib modules such as `logging`, `argparse`, `subprocess`, `json`, and `runpy` underpin `project/utils/logger.py` and `scripts/agent/*.py`.

## Configuration

**Environment:**
- `POINTCEPT_ROOT` is required by `scripts/train/train.py` and the smoke helpers in `scripts/check_data/`.
- `SBF_DATA_ROOT` is required by `configs/bf/semseg-pt-v3m1-0-base-bf.py` and `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-data.py`.
- No `.env` files were detected in the repository scan.

**Build:**
- Runtime config is expressed as Python dictionaries in `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`, related smoke/train variants in `configs/semantic_boundary/`, and dataset config in `configs/bf/semseg-pt-v3m1-0-base-bf.py`.
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-model.py` defines the PT-v3 backbone shape and edge-head channel contract.
- `README.md`, `install.md`, and `train.md` are the repository’s install/runtime guides; no separate build system is present.

## Platform Requirements

**Development:**
- A local workstation with Conda, Python, and an external Pointcept checkout reachable through `POINTCEPT_ROOT`.
- Filesystem access to a BF dataset root containing `training/` and `validation/` scene folders as documented in `docs/data_format.md`.
- CUDA-capable hardware is recommended for full training; CPU is mainly viable for smoke/debug validation.

**Production:**
- Not a deployed service. This repository is executed as local research/training code and writes artifacts into `outputs/` and `reports/`.
- No Dockerfile, packaged distribution, or CI deployment target was detected.

---

*Stack analysis: 2026-04-01*
