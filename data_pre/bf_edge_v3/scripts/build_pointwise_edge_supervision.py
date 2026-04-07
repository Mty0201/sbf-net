import argparse
from pathlib import Path

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()

from core.config import Stage4Config
from core.validation import validate_edge_supervision
from core.pointwise_core import (
    build_pointwise_edge_supervision,
    export_edge_arrays,
    export_edge_supervision_xyz,
    load_supports,
)
from utils.stage_io import load_scene


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pointwise edge supervision."""
    parser = argparse.ArgumentParser(description="BF Edge v3: build pointwise edge supervision")
    parser.add_argument("--input", type=str, required=True, help="Directory containing coord.npy, segment.npy, supports.npz")
    parser.add_argument("--output", type=str, default=None, help="Output directory. Default: same as input")
    parser.add_argument(
        "--support-radius",
        type=float,
        default=0.08,
        help="Support supervision radius for boundary snapping",
    )
    parser.add_argument("--ignore-index", type=int, default=-1, help="Semantic ignore label")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage4Config:
    """Build Stage4Config from CLI arguments."""
    return Stage4Config(
        support_radius=float(args.support_radius),
        ignore_index=int(args.ignore_index),
    )


def run_scene(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Run pointwise supervision generation for one stage directory."""
    cfg = build_config(args)
    scene = load_scene(input_dir)
    supports = load_supports(input_dir)
    payload, meta = build_pointwise_edge_supervision(
        scene=scene,
        supports=supports,
        support_radius=cfg.support_radius,
        ignore_index=cfg.ignore_index,
    )

    validate_edge_supervision(payload, num_scene_points=scene["coord"].shape[0])
    export_edge_arrays(output_dir=output_dir, payload=payload)
    export_edge_supervision_xyz(scene=scene, payload=payload, output_dir=output_dir)

    print("=" * 70)
    print("BF Edge v3 - build_pointwise_edge_supervision")
    print(f"  input: {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  points: {meta['num_points']}")
    print(f"  supports: {meta['num_supports']}")
    print(f"  valid_points: {meta['num_valid_points']}")
    print(f"  invalid_points: {meta['num_invalid_points']}")
    print(f"  support_radius: {meta['support_radius']:.6f}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output is not None else input_dir
    run_scene(input_dir=input_dir, output_dir=output_dir, args=args)


if __name__ == "__main__":
    main()
