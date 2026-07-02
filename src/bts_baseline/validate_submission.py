from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .dataset import find_scenes, image_files, read_test_poses
from .image_utils import get_image_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rendered folder against test_poses.csv.")
    parser.add_argument("--data-root", nargs="+", required=True)
    parser.add_argument("--pred-dir", required=True)
    args = parser.parse_args()

    scenes = find_scenes(args.data_root)
    pred_dir = Path(args.pred_dir)
    errors: list[str] = []

    expected_scenes = {scene.name for scene in scenes}
    actual_scenes = {path.name for path in pred_dir.iterdir() if path.is_dir()} if pred_dir.exists() else set()
    missing_scenes = expected_scenes - actual_scenes
    extra_scenes = actual_scenes - expected_scenes
    if missing_scenes:
        errors.append(f"Missing scene directories: {sorted(missing_scenes)}")
    if extra_scenes:
        errors.append(f"Extra scene directories: {sorted(extra_scenes)}")

    checked = 0
    for scene in scenes:
        scene_pred = pred_dir / scene.name
        test_poses = read_test_poses(scene.test_poses_csv)
        expected_files = {pose.image_name: pose for pose in test_poses}
        actual_files = {path.name for path in image_files(scene_pred)}

        missing_files = set(expected_files) - actual_files
        extra_files = actual_files - set(expected_files)
        if missing_files:
            errors.append(f"{scene.name}: missing files: {sorted(missing_files)[:10]}")
        if extra_files:
            errors.append(f"{scene.name}: extra files: {sorted(extra_files)[:10]}")

        for image_name, pose in expected_files.items():
            pred_path = scene_pred / image_name
            if not pred_path.is_file():
                continue
            try:
                size = get_image_size(pred_path)
            except Exception as exc:
                errors.append(f"{scene.name}/{image_name}: cannot read size: {exc}")
                continue
            expected_size = (pose.width, pose.height)
            if size != expected_size:
                errors.append(f"{scene.name}/{image_name}: size {size}, expected {expected_size}")
            checked += 1

    if errors:
        print("Validation failed:")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors")
        sys.exit(1)

    print(f"Validation OK: {len(scenes)} scenes, {checked} images.")


if __name__ == "__main__":
    main()
