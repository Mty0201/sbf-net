# External Integrations

**Analysis Date:** 2026-04-01

## APIs & External Services

**Upstream ML Codebase:**
- Pointcept - required external upstream repository that supplies the PT-v3 backbone, model and dataset registries, transform stack, point structure utilities, collate helpers, and Lovasz loss.
  - Integration method: local source checkout injected into `sys.path` by `scripts/train/train.py` and the smoke helpers in `scripts/check_data/`.
  - Auth: none. The integration is path-based through `POINTCEPT_ROOT` or `--pointcept-root`.
  - Touchpoints: `project/datasets/bf.py`, `project/models/semantic_boundary_model.py`, `project/models/semantic_model.py`, `project/trainer/trainer.py`, and `project/losses/*.py`.

**Remote APIs:**
- None detected.
  - Search scope: `project/`, `scripts/`, `configs/`, and `data_pre/`.
  - No HTTP client libraries, cloud SDKs, or webhook endpoints were found.

## Data Storage

**Databases:**
- None.
  - No SQL/NoSQL client libraries, migrations, or connection settings were detected.

**File Storage:**
- Local filesystem only.
  - Training data root is provided by `SBF_DATA_ROOT` in `configs/bf/semseg-pt-v3m1-0-base-bf.py` and `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-data.py`.
  - Expected dataset layout is documented in `docs/data_format.md` and loaded by `project/datasets/bf.py` and `data_pre/bf_edge_v3/utils/stage_io.py`.
  - Each scene is represented by local array files such as `coord.npy`, `color.npy`, `normal.npy`, `segment.npy`, `edge.npy`, and optional `edge_support_id.npy`.
  - Training artifacts are written under `outputs/` by `project/trainer/trainer.py` as `train.log`, `model_last.pth`, and `model_best.pth`.
  - Derived analysis artifacts are written under `reports/log_summaries/`, `reports/context_packets/`, `reports/round_updates/`, and `reports/workflow_smokes/` by `scripts/agent/summarize_train_log.py`, `scripts/agent/build_context_packet.py`, `scripts/agent/update_round_artifacts.py`, and `scripts/agent/workflow_consistency_smoke.py`.
  - Preprocessing scripts under `data_pre/bf_edge_v3/scripts/` read and write local `.npy`, `.npz`, and `.xyz` files in place.

**Caching:**
- None detected.
  - The repo reuses filesystem artifacts instead of a cache service.

## Authentication & Identity

**Auth Provider:**
- None.

**Implementation:**
- The only required environment variables are filesystem paths, not credentials: `POINTCEPT_ROOT` and `SBF_DATA_ROOT`.

**Session management:**
- Not applicable. This repository is not a user-facing service.

## Monitoring & Observability

**Error Tracking:**
- None detected.
  - No Sentry, Rollbar, or similar SDKs were found.

**Logs:**
- Local Python logging only.
  - `project/utils/logger.py` writes to stdout and to a local `train.log`.
  - `scripts/agent/summarize_train_log.py` parses those logs into markdown and JSON summaries for later review.

**Metrics:**
- Validation metrics stay in `train.log` and the derived summary files under `reports/log_summaries/`.
  - No external metrics backend, experiment tracker, or dashboard integration was detected.

## CI/CD & Deployment

**Hosting:**
- Not applicable.
  - The repository is intended for manual local execution, not for long-running hosted deployment.

**CI Pipeline:**
- None detected.
  - No `.github/workflows/`, `tox.ini`, `pytest.ini`, Dockerfiles, or deployment manifests were found.

**Execution model:**
- Manual CLI runs documented in `README.md`, `install.md`, and `train.md`, typically through `conda run --no-capture-output -n ptv3 python ...`.

## Environment Configuration

**Required env vars:**
- `POINTCEPT_ROOT` - external Pointcept checkout path used by `scripts/train/train.py` and `scripts/check_data/check_model_forward.py`.
- `SBF_DATA_ROOT` - BF dataset root used by `configs/bf/semseg-pt-v3m1-0-base-bf.py` and `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-data.py`.

**Secrets location:**
- Not applicable.
  - No `.env` files were detected, and no secret manager integration is documented in the repository.

**Mock/stub services:**
- Local smoke paths only.
  - `configs/semantic_boundary/*smoke.py` and the helpers in `scripts/check_data/` act as local verification entrypoints instead of networked stubs.

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None.

---

*Integration audit: 2026-04-01*
