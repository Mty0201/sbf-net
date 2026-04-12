"""Restructure s3dis from Area-based layout to training/validation splits.

Input:  <root>/Area_{1..6}/<room>/{coord,segment,normal,color,instance}.npy
Output: <root>/training/Area_{N}_{room}/{...}.npy   (Areas 1,2,3,4,6)
        <root>/validation/Area_5_{room}/{...}.npy    (Area 5)

Uses symlinks by default (--copy to do a full copy instead).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    train_dir = root / "training"
    val_dir = root / "validation"

    if not args.dry_run:
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

    train_areas = [1, 2, 3, 4, 6]
    val_areas = [5]

    total = 0
    for area_num in range(1, 7):
        area_dir = root / f"Area_{area_num}"
        if not area_dir.is_dir():
            print(f"WARN: {area_dir} not found, skipping")
            continue

        split = "validation" if area_num in val_areas else "training"
        out_base = val_dir if area_num in val_areas else train_dir

        rooms = sorted([p for p in area_dir.iterdir() if p.is_dir()])
        for room_dir in rooms:
            dest_name = f"Area_{area_num}_{room_dir.name}"
            dest = out_base / dest_name

            if dest.exists():
                print(f"  EXISTS {dest.relative_to(root)}")
                total += 1
                continue

            if args.dry_run:
                print(f"  [DRY] {split}/{dest_name} <- {room_dir.relative_to(root)}")
            elif args.copy:
                shutil.copytree(room_dir, dest)
                print(f"  COPY  {split}/{dest_name}")
            else:
                dest.symlink_to(room_dir)
                print(f"  LINK  {split}/{dest_name} -> {room_dir}")

            total += 1

    print(f"\nTotal: {total} rooms")


if __name__ == "__main__":
    main()
