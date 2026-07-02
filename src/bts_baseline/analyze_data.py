from __future__ import annotations

import argparse
from pathlib import Path

from .colmap_io import read_cameras_binary, read_images_binary, camera_center
from .dataset import find_scenes, image_files, read_test_poses
from .image_utils import get_image_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze contest data layout and COLMAP metadata.")
    parser.add_argument("--data-root", nargs="+", required=True)
    parser.add_argument("--output", default="reports/data_report.md")
    args = parser.parse_args()

    scenes = find_scenes(args.data_root)
    if not scenes:
        raise SystemExit(f"No scenes found under: {args.data_root}")

    lines: list[str] = [
        "# Data Report",
        "",
        f"Scenes found: **{len(scenes)}**",
        "",
        "| Scene | Root | Train images | Test poses | Test GT | Sparse images | Train registered | Test registered | Image size | CSV sizes | Camera |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]

    for scene in scenes:
        train_files = image_files(scene.train_images_dir)
        test_files = image_files(scene.test_images_dir)
        test_poses = read_test_poses(scene.test_poses_csv)
        sparse_images = read_images_binary(scene.sparse_dir / "images.bin")
        cameras = read_cameras_binary(scene.sparse_dir / "cameras.bin")

        train_names = {path.name for path in train_files}
        test_names = {pose.image_name for pose in test_poses}
        train_registered = sum(1 for image in sparse_images.values() if image.name in train_names)
        test_registered = sum(1 for image in sparse_images.values() if image.name in test_names)

        image_size = get_image_size(train_files[0]) if train_files else None
        csv_sizes = sorted({(pose.width, pose.height) for pose in test_poses})
        camera_desc = ", ".join(
            f"{camera.model} {camera.width}x{camera.height}"
            for camera in cameras.values()
        )
        rel_root = scene.path.parent.as_posix()
        lines.append(
            f"| {scene.name} | `{rel_root}` | {len(train_files)} | {len(test_poses)} | "
            f"{len(test_files)} | {len(sparse_images)} | {train_registered} | {test_registered} | "
            f"{image_size} | {csv_sizes} | {camera_desc} |"
        )

    lines.extend(["", "## Pose Notes", ""])
    lines.append(
        "`test_poses.csv` values `tx,ty,tz` align with COLMAP world-to-camera translation "
        "`tvec`. To compare camera locations, convert to center `C = -R^T t`."
    )

    for scene in scenes[:3]:
        sparse_images = read_images_binary(scene.sparse_dir / "images.bin")
        centers = [camera_center(image.qvec, image.tvec) for image in sparse_images.values()]
        center_stats = _stats(centers)
        lines.append(f"- `{scene.name}` sparse camera center ranges: {center_stats}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {output}")


def _stats(values: list[tuple[float, float, float]]) -> str:
    result = []
    for axis in zip(*values):
        result.append(f"{min(axis):.3f}..{max(axis):.3f}")
    return "(" + ", ".join(result) + ")"


if __name__ == "__main__":
    main()
