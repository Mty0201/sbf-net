# SBF-Net

**Semantic Boundary Field Network for Edge-Aware LiDAR Semantic Segmentation**

SBF-Net is a lightweight research project built on top of Pointcept for LiDAR point cloud semantic segmentation with semantic boundary field supervision. It keeps the semantic segmentation baseline while adding an edge-aware boundary field branch for point-wise boundary learning.

## Highlights

- Shared PTv3 backbone for joint feature extraction
- Semantic segmentation and boundary field dual-task design
- Boundary field supervision with mask-gated edge regression
- Non-intrusive integration with Pointcept as an upstream dependency

## Dependency on Pointcept

SBF-Net depends on Pointcept.

- This repository does **not** include the PTv3 implementation
- This repository does **not** vendor Pointcept source code
- You must prepare Pointcept first, then run SBF-Net with Pointcept as the upstream dependency

## Quick Start

1. Clone Pointcept and prepare its environment.
2. Activate the `ptv3` environment.
3. Clone SBF-Net.
4. Set:
   - `POINTCEPT_ROOT`
   - `SBF_DATA_ROOT`
5. Run training.

Example:

```bash
git clone <pointcept_repo_url> Pointcept
git clone <sbf_net_repo_url> SBF-Net

cd SBF-Net
conda activate ptv3

export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy

python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py \
  --pointcept-root "$POINTCEPT_ROOT"
```

Detailed setup and training instructions:

- [Install Guide](docs/install.md)
- [Training Guide](docs/train.md)
- [Project Structure](docs/project_structure.md)

## Training Commands

Smoke training:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train-smoke.py \
  --pointcept-root "$POINTCEPT_ROOT"
```

Full training:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py \
  --pointcept-root "$POINTCEPT_ROOT"
```

## Project Structure

Short version:

- `configs/`: smoke and full training configs
- `project/`: project-local datasets, models, losses, evaluator, trainer
- `scripts/`: runnable training and checking scripts
- `docs/`: project documentation

Detailed structure notes are in [project_structure.md](docs/project_structure.md).

## Current Status

Current status: **stage-1 trainable skeleton**

Implemented:

- BF base-field data path
- external edge synchronization path
- shared-backbone dual-head model shell
- minimal loss
- minimal evaluator
- project-local trainer
- validation metrics
- smoke and full training configs

Not implemented yet:

- test pipeline
- prediction export
- visualization export
- advanced hook system

## Roadmap

- Stabilize stage-1 full training on real GPU runs
- Add project-local validation integration refinement
- Design and implement test pipeline
- Refine edge-specific evaluation and reporting
- Prepare paper-style benchmark and ablation outputs

## Citation

Under preparation.

If you use SBF-Net in academic work before the citation entry is released, please cite the upstream Pointcept project and mention this repository in your implementation details.
