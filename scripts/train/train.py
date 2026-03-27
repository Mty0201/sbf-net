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
    pointcept_root_input = pointcept_root_arg or os.environ.get("POINTCEPT_ROOT")
    if pointcept_root_input is None:
        raise RuntimeError(
            "POINTCEPT_ROOT or --pointcept-root is required; implicit parent-directory fallback has been removed."
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
        help="Path to the Pointcept repository root. If omitted, use POINTCEPT_ROOT.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint path in config or model_last.pth in work_dir.",
    )
    parser.add_argument(
        "--weight",
        default=None,
        help="Load model weight or checkpoint path explicitly.",
    )
    return parser.parse_args()


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    args = parse_args(repo_root)
    repo_root, pointcept_root = bootstrap_paths(args.pointcept_root)

    from project.trainer import SemanticBoundaryTrainer

    cfg = runpy.run_path(args.config)
    cfg["resume"] = bool(args.resume or cfg.get("resume", False))
    if args.weight is not None:
        cfg["weight"] = args.weight
    print("env: ptv3")
    print(f"config: {args.config}")
    print(f"pointcept_root: {pointcept_root}")
    print(f"resume: {cfg.get('resume')}")
    print(f"weight: {cfg.get('weight')}")

    trainer = SemanticBoundaryTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
