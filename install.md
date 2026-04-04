# Install

This file is a runtime setup reference. Default planning and execution for this repository start with GSD and `.planning/`, not with the install guide. For repository-specific rules and current technical status, read [docs/canonical/README.md](docs/canonical/README.md).

## 1. Pointcept Is Required

`sbf-net` is not a standalone replacement for Pointcept.

Prepare Pointcept first. This repository reuses Pointcept for:

- PTv3 backbone implementation
- point structure utilities
- registry and runtime interfaces

The maintenance boundary stays the same: this repository contains the SBF-specific code, while Pointcept remains an external host dependency.

## 2. Recommended Layout

```text
workspace/
├── Pointcept/
└── sbf-net/
```

If the repositories are not side by side, pass `--pointcept-root /path/to/Pointcept` when running commands.

## 3. Environment

Use the same Conda environment as Pointcept.

Recommended environment name:

```bash
ptv3
```

Activate it with:

```bash
conda activate ptv3
```

Or run commands directly with:

```bash
conda run --no-capture-output -n ptv3 <command>
```

## 4. Required Inputs

Set the Pointcept root:

```bash
export POINTCEPT_ROOT=/path/to/Pointcept
```

Set the BF dataset root:

```bash
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
```

Expected BF split names:

- `training`
- `validation`

Dataset format details remain in [docs/data_format.md](docs/data_format.md).

## 5. Minimal Verification

After activating the environment and setting variables, verify the setup with:

```bash
conda run --no-capture-output -n ptv3 python -c "import os, torch; print('torch_ok', torch.__version__); print('POINTCEPT_ROOT', os.environ.get('POINTCEPT_ROOT')); print('SBF_DATA_ROOT', os.environ.get('SBF_DATA_ROOT'))"
```

If you also want to confirm that the Pointcept path is usable:

```bash
conda run --no-capture-output -n ptv3 python -c "import os, sys; sys.path.insert(0, os.environ['POINTCEPT_ROOT']); import pointcept; print('pointcept_ok')"
```

## 6. Next Reference

Once the environment is ready:

- go back to GSD and `.planning/` for workflow control
- read [train.md](train.md) for runtime command patterns
- read [docs/canonical/sbf_training_guardrails.md](docs/canonical/sbf_training_guardrails.md) for the canonical no-fallback training rules
