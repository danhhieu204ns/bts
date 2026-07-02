from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

from .colmap_io import ImagePose, read_images_binary


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Scene:
    name: str
    path: Path

    @property
    def train_images_dir(self) -> Path:
        return self.path / "train" / "images"

    @property
    def sparse_dir(self) -> Path:
        return self.path / "train" / "sparse" / "0"

    @property
    def test_poses_csv(self) -> Path:
        return self.path / "test" / "test_poses.csv"

    @property
    def test_images_dir(self) -> Path:
        return self.path / "test" / "images"


@dataclass(frozen=True)
class TestPose:
    image_name: str
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def find_scenes(data_roots: list[str | Path]) -> list[Scene]:
    scenes: dict[Path, Scene] = {}
    for root_value in data_roots:
        root = Path(root_value)
        if _is_scene_dir(root):
            scenes[root.resolve()] = Scene(root.name, root)
            continue

        for csv_path in root.rglob("test_poses.csv"):
            scene_path = csv_path.parent.parent
            if _is_scene_dir(scene_path):
                scenes[scene_path.resolve()] = Scene(scene_path.name, scene_path)

    return sorted(scenes.values(), key=lambda scene: scene.name.lower())


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_test_poses(csv_path: str | Path) -> list[TestPose]:
    rows: list[TestPose] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fid:
        reader = csv.DictReader(fid)
        required = {
            "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "fx", "fy", "cx", "cy", "width", "height",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            rows.append(
                TestPose(
                    image_name=row["image_name"],
                    qvec=(float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])),
                    tvec=(float(row["tx"]), float(row["ty"]), float(row["tz"])),
                    fx=float(row["fx"]),
                    fy=float(row["fy"]),
                    cx=float(row["cx"]),
                    cy=float(row["cy"]),
                    width=int(float(row["width"])),
                    height=int(float(row["height"])),
                )
            )
    return rows


def load_registered_train_poses(scene: Scene) -> dict[str, ImagePose]:
    sparse_images = read_images_binary(scene.sparse_dir / "images.bin")
    train_names = {path.name for path in image_files(scene.train_images_dir)}
    return {
        image.name: image
        for image in sparse_images.values()
        if image.name in train_names
    }


def _is_scene_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "train" / "images").is_dir()
        and (path / "train" / "sparse" / "0" / "images.bin").is_file()
        and (path / "train" / "sparse" / "0" / "cameras.bin").is_file()
        and (path / "test" / "test_poses.csv").is_file()
    )
