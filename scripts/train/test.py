"""Project-local test entry for semantic boundary evaluation.

Usage:
    POINTCEPT_ROOT=/path/to/Pointcept python scripts/train/test.py \
        --config configs/semantic_boundary/clean_reset_s38873367/semantic_only_train.py \
        --weight outputs/clean_reset_s38873367/semantic_only/model/model_best.pth
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def bootstrap_paths(pointcept_root_arg: str | None = None) -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    pointcept_root_input = pointcept_root_arg or os.environ.get("POINTCEPT_ROOT")
    if pointcept_root_input is None:
        raise RuntimeError(
            "POINTCEPT_ROOT or --pointcept-root is required."
        )
    pointcept_root = Path(pointcept_root_input).resolve()
    if not pointcept_root.exists():
        raise FileNotFoundError(f"Pointcept root not found: {pointcept_root}")
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root, pointcept_root


def parse_args(repo_root: Path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--pointcept-root",
        default=None,
        help="Path to the Pointcept repository root. If omitted, use POINTCEPT_ROOT.",
    )
    parser.add_argument(
        "--weight",
        required=True,
        help="Path to the model checkpoint (.pth) to evaluate.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers for test.",
    )
    return parser.parse_args()


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    args = parse_args(repo_root)
    repo_root, pointcept_root = bootstrap_paths(args.pointcept_root)

    from project.tester import SemanticBoundaryTester

    cfg = runpy.run_path(args.config)
    cfg["weight"] = args.weight
    cfg["test_num_workers"] = args.num_workers

    print("env: ptv3")
    print(f"config: {args.config}")
    print(f"pointcept_root: {pointcept_root}")
    print(f"weight: {args.weight}")

    tester = SemanticBoundaryTester(cfg)
    tester.test()


if __name__ == "__main__":
    main()
