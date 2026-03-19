# Project Structure

## Overview

SBF-Net is organized as a small independent repository that sits on top of Pointcept. The repository keeps Pointcept as an upstream dependency and concentrates project-specific code and documentation locally.

## Top-Level Directories

- `configs/`
  Project-local configs, including smoke training and full training variants.

- `docs/`
  Project documentation, design notes, install guide, training guide, and collaboration rules.

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
  Project-local minimal trainer used by the training entry script.

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

- `install.md`
- `train.md`
- `data_format.md`
- `validation_design.md`
- `research_plan.md`
- `pointcept_boundary.md`
- `agents.md`

## Current Scope

Current repository structure is intentionally minimal. It is designed to support:

- stage-1 training
- stage-1 validation
- smoke verification

It does not yet provide:

- test pipeline
- visualization export
- benchmark packaging
- multi-dataset support
