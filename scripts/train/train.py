"""Project-local minimal training entry for semantic boundary training."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def bootstrap_paths(pointcept_root_arg: str | None = None) -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    pointcept_root = (
        Path(pointcept_root_arg).resolve()
        if pointcept_root_arg is not None
        else Path(os.environ.get("POINTCEPT_ROOT", repo_root.parent)).resolve()
    )
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root, pointcept_root


def parse_args(repo_root: Path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            repo_root
            / "configs"
            / "semantic_boundary"
            / "semseg-pt-v3m1-0-base-bf-edge-train.py"
        ),
    )
    parser.add_argument(
        "--pointcept-root",
        default=None,
        help="Path to the Pointcept repository root. "
        "If omitted, use POINTCEPT_ROOT or the parent directory of this repo.",
    )
    return parser.parse_args()


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    args = parse_args(repo_root)
    repo_root, pointcept_root = bootstrap_paths(args.pointcept_root)

    from project.trainer import SemanticBoundaryTrainer

    cfg = runpy.run_path(args.config)
    print("env: ptv3")
    print(f"config: {args.config}")
    print(f"pointcept_root: {pointcept_root}")

    trainer = SemanticBoundaryTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
