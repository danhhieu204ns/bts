from __future__ import annotations

import argparse
import csv
from pathlib import Path
import zipfile

from .dataset import Scene, find_scenes, load_registered_train_poses, read_test_poses
from .image_utils import copy_or_resize_image
from .pose import pose_distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest-pose image-copy baseline.")
    parser.add_argument("--data-root", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--orientation-weight", type=float, default=0.5)
    parser.add_argument("--zip", dest="zip_path", default="")
    args = parser.parse_args()

    scenes = find_scenes(args.data_root)
    if not scenes:
        raise SystemExit(f"No scenes found under: {args.data_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    for scene in scenes:
        rows = run_scene(scene, out_dir, args.orientation_weight)
        manifest_rows.extend(rows)
        print(f"{scene.name}: rendered {len(rows)} images")

    manifest_path = out_dir / "nearest_pose_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"Wrote manifest: {manifest_path}")

    if args.zip_path:
        zip_submission(out_dir, Path(args.zip_path), scenes)
        print(f"Wrote zip: {args.zip_path}")


def run_scene(scene: Scene, out_dir: Path, orientation_weight: float) -> list[dict[str, str]]:
    train_poses = load_registered_train_poses(scene)
    if not train_poses:
        raise RuntimeError(f"{scene.name}: no registered train poses found")

    test_poses = read_test_poses(scene.test_poses_csv)
    scene_out = out_dir / scene.name
    scene_out.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    train_items = sorted(train_poses.items())
    for test_pose in test_poses:
        best_name = ""
        best_score = float("inf")
        best_center_distance = float("inf")
        best_angle = float("inf")

        for train_name, train_pose in train_items:
            score, center_distance, angle = pose_distance(
                train_pose.qvec,
                train_pose.tvec,
                test_pose.qvec,
                test_pose.tvec,
                orientation_weight,
            )
            if score < best_score:
                best_name = train_name
                best_score = score
                best_center_distance = center_distance
                best_angle = angle

        src = scene.train_images_dir / best_name
        dst = scene_out / test_pose.image_name
        copy_or_resize_image(src, dst, (test_pose.width, test_pose.height))
        manifest_rows.append(
            {
                "scene": scene.name,
                "image_name": test_pose.image_name,
                "selected_train_image": best_name,
                "score": f"{best_score:.8f}",
                "center_distance": f"{best_center_distance:.8f}",
                "angle_rad": f"{best_angle:.8f}",
            }
        )

    return manifest_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def zip_submission(out_dir: Path, zip_path: Path, scenes: list[Scene]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    scene_names = {scene.name for scene in scenes}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for scene_name in sorted(scene_names, key=str.lower):
            scene_dir = out_dir / scene_name
            for image_path in sorted(scene_dir.iterdir(), key=lambda path: path.name.lower()):
                if image_path.is_file():
                    archive.write(image_path, image_path.relative_to(out_dir).as_posix())


if __name__ == "__main__":
    main()
