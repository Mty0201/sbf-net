# Project Structure

## Overview

SBF-Net is organized as a small independent repository that sits on top of Pointcept. The repository keeps Pointcept as an upstream dependency and concentrates project-specific code and documentation locally.

## Top-Level Directories

- `configs/`
  Project-local configs, including smoke training and full training variants.

- `docs/`
  Internal design notes, boundary documents, and collaboration rules.

- `patches/`
  Reserved for minimal Pointcept patch notes if a future stage proves them unavoidable.

- `project/`
  Main project-local Python code.

- `scripts/`
  Runnable entry scripts for training and smoke checks.

## project/

- `project/datasets/`
  BF dataset wrapper and project-local data loading extensions.

- `project/transforms/`
  Project-local transform helpers, including edge synchronization support.

- `project/models/`
  Shared-backbone dual-head model shell and project-local heads.

- `project/losses/`
  Stage-1 semantic and boundary loss implementation.

- `project/evaluator/`
  Stage-1 validation metric computation.

- `project/trainer/`
  Project-local single-card trainer used by the training entry script.

## scripts/

- `scripts/train/`
  Training entry scripts.

- `scripts/check_data/`
  Smoke tests and local verification scripts for data, model, loss, and validation steps.

## configs/

- `configs/bf/`
  BF dataset-related config fragments.

- `configs/semantic_boundary/`
  Model and training configs for SBF-Net.

## docs/

Important documents:

- `data_format.md`
- `validation_design.md`
- `research_plan.md`
- `pointcept_boundary.md`
- `agents.md`

User-facing entry documents live at the repository root:

- `README.md`
- `install.md`
- `train.md`

## Current Scope

Current repository structure is intentionally minimal. It is designed to support:

- trainable semantic-plus-boundary learning
- project-local validation
- smoke verification
- ongoing runtime refinement

It does not yet provide:

- test pipeline
- visualization export
- benchmark packaging
- multi-dataset support
