from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

from .colmap_io import camera_to_pinhole, read_cameras_binary, read_images_binary, write_cameras_binary, write_images_binary
from .dataset import Scene, find_scenes, image_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare contest scenes for official 3D Gaussian Splatting.")
    parser.add_argument("--data-root", nargs="+", required=True)
    parser.add_argument("--out-root", default="prepared/3dgs_data")
    parser.add_argument("--scene", default="", help="Optional scene name filter.")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--camera-mode", choices=["pinhole", "copy"], default="pinhole")
    args = parser.parse_args()

    scenes = find_scenes(args.data_root)
    if args.scene:
        scene_names = {name.strip() for name in args.scene.split(",") if name.strip()}
        scenes = [scene for scene in scenes if scene.name in scene_names]
    if not scenes:
        raise SystemExit("No matching scenes found.")

    out_root = Path(args.out_root)
    for scene in scenes:
        result = prepare_scene(scene, out_root, args.copy_mode, args.camera_mode)
        print(
            f"{scene.name}: prepared {result['train_images']} train images, "
            f"filtered images.bin {result['sparse_before']} -> {result['sparse_after']}, "
            f"camera={result['camera_mode']} at {result['out_dir']}"
        )


def prepare_scene(scene: Scene, out_root: Path, copy_mode: str, camera_mode: str) -> dict[str, object]:
    out_dir = out_root / scene.name
    out_images = out_dir / "images"
    out_sparse = out_dir / "sparse" / "0"
    out_images.mkdir(parents=True, exist_ok=True)
    out_sparse.mkdir(parents=True, exist_ok=True)

    train_images = image_files(scene.train_images_dir)
    train_names = {path.name for path in train_images}
    for image_path in train_images:
        link_or_copy(image_path, out_images / image_path.name, copy_mode)

    for sparse_file in scene.sparse_dir.glob("*"):
        if sparse_file.is_file() and sparse_file.name != "images.bin":
            shutil.copy2(sparse_file, out_sparse / sparse_file.name)

    if camera_mode == "pinhole":
        original_cameras = read_cameras_binary(scene.sparse_dir / "cameras.bin")
        converted_cameras = {
            camera_id: camera_to_pinhole(camera)
            for camera_id, camera in original_cameras.items()
        }
        shutil.copy2(scene.sparse_dir / "cameras.bin", out_sparse / "cameras_original.bin")
        write_cameras_binary(converted_cameras, out_sparse / "cameras.bin")

    sparse_images = read_images_binary(scene.sparse_dir / "images.bin")
    filtered = {
        image_id: image
        for image_id, image in sparse_images.items()
        if image.name in train_names
    }
    write_images_binary(filtered, out_sparse / "images.bin")

    missing_registered = train_names - {image.name for image in filtered.values()}
    if missing_registered:
        missing_preview = ", ".join(sorted(missing_registered)[:5])
        raise RuntimeError(f"{scene.name}: train images missing in COLMAP images.bin: {missing_preview}")

    return {
        "out_dir": out_dir,
        "train_images": len(train_images),
        "sparse_before": len(sparse_images),
        "sparse_after": len(filtered),
        "camera_mode": camera_mode,
    }


def link_or_copy(src: Path, dst: Path, copy_mode: str) -> None:
    if dst.exists():
        return
    if copy_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
