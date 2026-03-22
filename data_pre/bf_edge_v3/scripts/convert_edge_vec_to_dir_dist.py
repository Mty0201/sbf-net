import argparse
from pathlib import Path

import numpy as np


SPLITS = ("training", "validation")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for edge.npy vec -> dir+dist conversion."""
    parser = argparse.ArgumentParser(
        description="BF Edge v3: convert edge.npy from vec/support/valid to dir/dist/support/valid"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input legacy edge.npy path, one scene directory containing edge.npy, or dataset root with training/validation scenes",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output converted edge.npy path, one output scene directory, or output dataset root",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Direction normalization threshold. Direction is zeroed when dist < eps.",
    )
    return parser.parse_args()


def collect_scene_dirs(dataset_root: Path) -> list[Path]:
    """Collect scene directories containing edge.npy under training/validation."""
    scene_dirs: list[Path] = []
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if (scene_dir / "edge.npy").exists():
                scene_dirs.append(scene_dir)
    return scene_dirs


def load_legacy_edge(path: Path) -> np.ndarray:
    """Load and validate legacy edge.npy."""
    edge = np.load(path)
    if edge.ndim != 2 or edge.shape[1] != 5:
        raise ValueError(
            f"Expected legacy edge.npy with shape (N, 5), but got {edge.shape}."
        )
    return edge.astype(np.float32, copy=False)


def convert_edge_array(edge: np.ndarray, eps: float) -> np.ndarray:
    """Convert [vec_x, vec_y, vec_z, support, valid] to [dir_x, dir_y, dir_z, dist, support, valid]."""
    if eps <= 0:
        raise ValueError(f"eps must be positive, but got {eps}.")

    vec = edge[:, 0:3].astype(np.float32, copy=False)
    support = edge[:, 3].astype(np.float32, copy=False)
    valid = edge[:, 4].astype(np.float32, copy=False)

    dist = np.linalg.norm(vec, axis=1).astype(np.float32)
    direction = np.zeros_like(vec, dtype=np.float32)

    safe_mask = dist >= float(eps)
    if np.any(safe_mask):
        direction[safe_mask] = vec[safe_mask] / dist[safe_mask, None]

    converted = np.zeros((edge.shape[0], 6), dtype=np.float32)
    converted[:, 0:3] = direction
    converted[:, 3] = dist
    converted[:, 4] = support
    converted[:, 5] = valid
    return converted


def convert_one_file(input_path: Path, output_path: Path, eps: float) -> None:
    """Convert one legacy edge.npy file."""
    edge = load_legacy_edge(input_path)
    converted = convert_edge_array(edge=edge, eps=eps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, converted)

    dist = converted[:, 3]
    zero_dir_count = int(np.count_nonzero(dist < float(eps)))

    print("=" * 70)
    print("BF Edge v3 - convert_edge_vec_to_dir_dist")
    print(f"  input: {input_path}")
    print(f"  output: {output_path}")
    print(f"  input_shape: {tuple(edge.shape)}")
    print(f"  output_shape: {tuple(converted.shape)}")
    print(f"  eps: {float(eps):.3e}")
    print(f"  zero_direction_points: {zero_dir_count}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    eps = float(args.eps)

    if input_path.is_file():
        convert_one_file(input_path=input_path, output_path=output_path, eps=eps)
        return

    if (input_path / "edge.npy").exists():
        convert_one_file(
            input_path=input_path / "edge.npy",
            output_path=output_path / "edge.npy" if output_path.suffix != ".npy" else output_path,
            eps=eps,
        )
        return

    scene_dirs = collect_scene_dirs(input_path)
    if not scene_dirs:
        raise ValueError(
            "Input must be a legacy edge.npy file, a scene directory containing edge.npy, "
            "or a dataset root with training/validation scenes containing edge.npy."
        )

    for scene_dir in scene_dirs:
        rel_path = scene_dir.relative_to(input_path)
        convert_one_file(
            input_path=scene_dir / "edge.npy",
            output_path=output_path / rel_path / "edge.npy",
            eps=eps,
        )


if __name__ == "__main__":
    main()
