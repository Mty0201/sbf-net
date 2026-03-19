"""Smoke test for BF base-field dataset loading through Pointcept."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    pointcept_root = repo_root.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root


def describe_value(key, value):
    shape = tuple(value.shape) if hasattr(value, "shape") else None
    dtype = str(value.dtype) if hasattr(value, "dtype") else type(value).__name__
    print(f"  {key}: shape={shape}, dtype={dtype}")


def main():
    repo_root = bootstrap_paths()

    # Import project dataset package so BFDataset is registered into Pointcept.
    import project.datasets  # noqa: F401
    import project.transforms  # noqa: F401
    from pointcept.datasets import build_dataset

    cfg_path = repo_root / "configs" / "bf" / "semseg-pt-v3m1-0-base-bf.py"
    cfg = runpy.run_path(str(cfg_path))

    train_ds = build_dataset(cfg["data"]["train"])
    val_ds = build_dataset(cfg["data"]["val"])

    print(f"env: ptv3")
    print(f"config: {cfg_path}")
    print(f"train_len: {len(train_ds)}")
    print(f"val_len: {len(val_ds)}")

    raw_train = train_ds.get_data(0)
    train_sample = train_ds[0]
    val_sample = val_ds[0]

    print("raw_train_keys:", sorted(raw_train.keys()))
    for key in sorted(raw_train.keys()):
        describe_value(key, raw_train[key])
    print("raw_edge_in_sample:", "edge" in raw_train)
    print("raw_edge_shape:", tuple(raw_train["edge"].shape) if "edge" in raw_train else None)
    print(
        "raw_edge_aligned:",
        ("edge" in raw_train and raw_train["coord"].shape[0] == raw_train["edge"].shape[0]),
    )

    for name, sample in [("train", train_sample), ("val", val_sample)]:
        print(f"{name}_keys:", sorted(sample.keys()))
        for key in sorted(sample.keys()):
            describe_value(key, sample[key])
        print(f"{name}_edge_in_sample:", "edge" in sample)
        print(
            f"{name}_edge_shape:",
            tuple(sample["edge"].shape) if "edge" in sample else None,
        )
        print(
            f"{name}_edge_aligned:",
            ("edge" in sample and sample["coord"].shape[0] == sample["edge"].shape[0]),
        )


if __name__ == "__main__":
    main()
