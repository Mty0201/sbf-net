# Install

This document describes how to prepare SBF-Net on top of Pointcept.  
本文档说明如何在 Pointcept 之上准备 SBF-Net。

## 1. Pointcept Is Required

SBF-Net is not a standalone replacement for Pointcept.

You must prepare Pointcept first, because SBF-Net reuses Pointcept for:

- PTv3 backbone implementation
- point structure utilities
- dataset and transform registries

SBF-Net only provides the project-specific incremental code.

## 2. Recommended Setup

Recommended workspace layout:

```text
workspace/
├── Pointcept/
└── SBF-Net/
```

This layout is recommended but not mandatory. If the two repositories are not placed side by side, pass `--pointcept-root /path/to/Pointcept` when running SBF-Net commands.

A typical setup sequence is:

```bash
git clone <your-pointcept-repo> Pointcept
git clone <your-sbf-net-repo> SBF-Net
cd SBF-Net
conda activate ptv3
```

## 3. Conda Environment

Use the same environment as Pointcept.

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

## 4. Pointcept Installation Note

This document does not duplicate the full Pointcept installation guide.

Before using SBF-Net, make sure:

- Pointcept is available on disk
- Pointcept dependencies are already installed
- the Pointcept environment can import `torch`
- the Pointcept environment can import `pointcept`

## 5. Required Environment Variables

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

Dataset format details are in [docs/data_format.md](docs/data_format.md).

## 6. Optional Alternative To POINTCEPT_ROOT

Instead of exporting `POINTCEPT_ROOT`, you may pass it explicitly:

```bash
python scripts/train/train.py --pointcept-root /path/to/Pointcept
```

The same entry is used for:

- dual-task training
- semantic-only calibration training

## 7. Minimal Environment Verification

After activating the environment and setting variables, you can verify the setup with:

```bash
conda run --no-capture-output -n ptv3 python -c "import os, torch; print('torch_ok', torch.__version__); print('POINTCEPT_ROOT', os.environ.get('POINTCEPT_ROOT')); print('SBF_DATA_ROOT', os.environ.get('SBF_DATA_ROOT'))"
```

If you also want to confirm that the Pointcept path is usable:

```bash
conda run --no-capture-output -n ptv3 python -c "import os, sys; sys.path.insert(0, os.environ['POINTCEPT_ROOT']); import pointcept; print('pointcept_ok')"
```

## 8. What This Install Step Does Not Do

Current installation guidance does not include:

- `pip install -e .` packaging for SBF-Net
- test pipeline setup
- visualization tools
- multi-dataset support

The current goal is a clear, reproducible, trainable repository layout for the first public SBF-Net release.
