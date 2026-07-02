from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from .colmap_io import (
    Camera,
    ImagePose,
    camera_to_pinhole,
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from .dataset import Scene, TestPose, find_scenes, image_files, read_test_poses
from .pose import pose_distance


@dataclass(frozen=True)
class TrainedPointCloud:
    xyz: np.ndarray
    colors: np.ndarray
    num_observations: int
    num_fitted_points: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and render a COLMAP point-splat 3D baseline.")
    parser.add_argument("--data-root", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-dir", default="outputs/point_splat_models")
    parser.add_argument("--scene", default="", help="Optional comma-separated scene filter.")
    parser.add_argument("--background", choices=["nearest", "solid"], default="nearest")
    parser.add_argument("--solid-bg", nargs=3, type=int, default=[255, 255, 255])
    parser.add_argument("--splat-radius", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--max-points", type=int, default=0, help="Optional cap for fastest smoke tests.")
    parser.add_argument("--zip", dest="zip_path", default="")
    args = parser.parse_args()

    scenes = find_scenes(args.data_root)
    if args.scene:
        keep = {name.strip() for name in args.scene.split(",") if name.strip()}
        scenes = [scene for scene in scenes if scene.name in keep]
    if not scenes:
        raise SystemExit("No matching scenes found.")

    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    for scene in scenes:
        rows = run_scene(
            scene=scene,
            out_dir=out_dir,
            model_dir=model_dir,
            background=args.background,
            solid_bg=tuple(args.solid_bg),
            splat_radius=args.splat_radius,
            alpha=args.alpha,
            max_points=args.max_points,
        )
        all_rows.extend(rows)
        print(f"{scene.name}: rendered {len(rows)} images")

    manifest_path = out_dir / "point_splat_manifest.csv"
    write_manifest(manifest_path, all_rows)
    print(f"Wrote manifest: {manifest_path}")

    if args.zip_path:
        zip_submission(out_dir, Path(args.zip_path), scenes)
        print(f"Wrote zip: {args.zip_path}")


def run_scene(
    scene: Scene,
    out_dir: Path,
    model_dir: Path,
    background: str,
    solid_bg: tuple[int, int, int],
    splat_radius: int,
    alpha: float,
    max_points: int,
) -> list[dict[str, str]]:
    sparse_images = read_images_binary(scene.sparse_dir / "images.bin")
    cameras = read_cameras_binary(scene.sparse_dir / "cameras.bin")
    camera = camera_to_pinhole(next(iter(cameras.values())))
    train_names = {path.name for path in image_files(scene.train_images_dir)}
    train_images = {image_id: image for image_id, image in sparse_images.items() if image.name in train_names}
    test_poses = read_test_poses(scene.test_poses_csv)

    trained = train_point_cloud(scene, sparse_images, train_images, max_points)
    model_path = model_dir / f"{scene.name}.npz"
    np.savez_compressed(
        model_path,
        xyz=trained.xyz.astype(np.float32),
        colors=trained.colors.astype(np.float32),
        num_observations=np.array([trained.num_observations], dtype=np.int64),
        num_fitted_points=np.array([trained.num_fitted_points], dtype=np.int64),
    )
    print(
        f"{scene.name}: trained point cloud points={len(trained.xyz)} "
        f"fitted={trained.num_fitted_points} obs={trained.num_observations}"
    )

    scene_out = out_dir / scene.name
    scene_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    train_items = sorted(train_images.values(), key=lambda image: image.name)
    for pose in test_poses:
        bg_image, bg_name = make_background(scene, train_items, pose, background, solid_bg)
        rendered = render_point_splat(
            xyz=trained.xyz,
            colors=trained.colors,
            pose=pose,
            background=bg_image,
            splat_radius=splat_radius,
            alpha=alpha,
        )
        save_rgb(scene_out / pose.image_name, rendered)
        rows.append(
            {
                "scene": scene.name,
                "image_name": pose.image_name,
                "background": bg_name,
                "points": str(len(trained.xyz)),
                "fitted_points": str(trained.num_fitted_points),
                "observations": str(trained.num_observations),
            }
        )
    return rows


def train_point_cloud(
    scene: Scene,
    sparse_images: dict[int, ImagePose],
    train_images: dict[int, ImagePose],
    max_points: int,
) -> TrainedPointCloud:
    points = list(read_points3d_binary(scene.sparse_dir / "points3D.bin").values())
    points.sort(key=lambda point: point.point3d_id)
    if max_points > 0:
        points = points[:max_points]

    xyz = np.asarray([point.xyz for point in points], dtype=np.float32)
    base_colors = np.asarray([point.rgb for point in points], dtype=np.float32) / 255.0
    accum = np.zeros_like(base_colors, dtype=np.float64)
    counts = np.zeros((len(points),), dtype=np.float64)

    point_index = {point.point3d_id: idx for idx, point in enumerate(points)}
    observations_by_image: dict[int, list[tuple[int, int]]] = {}
    for point in points:
        idx = point_index[point.point3d_id]
        for image_id, point2d_idx in point.track:
            if image_id in train_images:
                observations_by_image.setdefault(image_id, []).append((idx, point2d_idx))

    num_observations = 0
    for image_id, observations in observations_by_image.items():
        image_pose = sparse_images[image_id]
        image_path = scene.train_images_dir / image_pose.name
        if not image_path.is_file():
            continue
        pixels = load_rgb_float(image_path)
        height, width = pixels.shape[:2]
        points2d = image_pose.points2d
        for point_idx, point2d_idx in observations:
            if point2d_idx < 0 or point2d_idx >= len(points2d):
                continue
            x, y, _point_id = points2d[point2d_idx]
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < width and 0 <= iy < height:
                accum[point_idx] += pixels[iy, ix]
                counts[point_idx] += 1.0
                num_observations += 1

    fitted_colors = base_colors.copy()
    fitted_mask = counts > 0
    fitted_colors[fitted_mask] = (accum[fitted_mask] / counts[fitted_mask, None]).astype(np.float32)
    return TrainedPointCloud(
        xyz=xyz,
        colors=np.clip(fitted_colors, 0.0, 1.0),
        num_observations=num_observations,
        num_fitted_points=int(fitted_mask.sum()),
    )


def render_point_splat(
    xyz: np.ndarray,
    colors: np.ndarray,
    pose: TestPose,
    background: np.ndarray,
    splat_radius: int,
    alpha: float,
) -> np.ndarray:
    rot = np.asarray(qvec2rotmat(pose.qvec), dtype=np.float32)
    trans = np.asarray(pose.tvec, dtype=np.float32)
    cam = xyz @ rot.T + trans[None, :]
    z = cam[:, 2]
    valid = z > 0.03
    if not np.any(valid):
        return background

    cam = cam[valid]
    z = z[valid]
    point_colors = colors[valid]
    u = pose.fx * (cam[:, 0] / z) + pose.cx
    v = pose.fy * (cam[:, 1] / z) + pose.cy
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < pose.width) & (vi >= 0) & (vi < pose.height)
    if not np.any(in_bounds):
        return background

    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = z[in_bounds]
    point_colors = point_colors[in_bounds]
    linear = vi.astype(np.int64) * pose.width + ui.astype(np.int64)
    order = np.lexsort((z, linear))
    linear_sorted = linear[order]
    first = np.empty_like(linear_sorted, dtype=bool)
    first[0] = True
    first[1:] = linear_sorted[1:] != linear_sorted[:-1]
    selected = order[first]

    overlay = np.zeros((pose.height, pose.width, 3), dtype=np.float32)
    mask = np.zeros((pose.height, pose.width, 1), dtype=np.float32)
    overlay[vi[selected], ui[selected]] = point_colors[selected]
    mask[vi[selected], ui[selected], 0] = 1.0

    if splat_radius > 0:
        kernel = (2 * splat_radius + 1, 2 * splat_radius + 1)
        color_sum = cv2.boxFilter(overlay, ddepth=-1, ksize=kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
        count = cv2.boxFilter(mask, ddepth=-1, ksize=kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
        count_2d = count if count.ndim == 2 else count[..., 0]
        composite_mask = count_2d > 0
        splat = np.zeros_like(overlay)
        splat[composite_mask] = color_sum[composite_mask] / count_2d[composite_mask, None]
    else:
        composite_mask = mask[..., 0] > 0
        splat = overlay

    result = background.astype(np.float32) / 255.0
    result[composite_mask] = (1.0 - alpha) * result[composite_mask] + alpha * splat[composite_mask]
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def make_background(
    scene: Scene,
    train_items: list[ImagePose],
    pose: TestPose,
    background: str,
    solid_bg: tuple[int, int, int],
) -> tuple[np.ndarray, str]:
    if background == "solid":
        bg = np.zeros((pose.height, pose.width, 3), dtype=np.uint8)
        bg[:, :] = np.asarray(solid_bg, dtype=np.uint8)
        return bg, "solid"

    best_pose = min(
        train_items,
        key=lambda train_pose: pose_distance(train_pose.qvec, train_pose.tvec, pose.qvec, pose.tvec, 0.5)[0],
    )
    bg = load_rgb_uint8(scene.train_images_dir / best_pose.name)
    if bg.shape[1] != pose.width or bg.shape[0] != pose.height:
        bg = cv2.resize(bg, (pose.width, pose.height), interpolation=cv2.INTER_LANCZOS4)
    return bg, best_pose.name


def load_rgb_float(path: Path) -> np.ndarray:
    return load_rgb_uint8(path).astype(np.float32) / 255.0


def load_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(image, mode="RGB")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        pil_image.save(path, quality=95)
    else:
        pil_image.save(path)


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
