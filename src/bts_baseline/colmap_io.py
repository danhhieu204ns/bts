from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import struct


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_NAME_TO_ID = {name: model_id for model_id, (name, _) in CAMERA_MODELS.items()}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    points2d: tuple[tuple[float, float, int], ...] = ()

    @property
    def center(self) -> tuple[float, float, float]:
        return camera_center(self.qvec, self.tvec)


@dataclass(frozen=True)
class Point3D:
    point3d_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: tuple[tuple[int, int], ...]


def read_cameras_binary(path: str | Path) -> dict[int, Camera]:
    path = Path(path)
    cameras: dict[int, Camera] = {}
    with path.open("rb") as fid:
        num_cameras = _unpack(fid, "Q")[0]
        for _ in range(num_cameras):
            camera_id = _unpack(fid, "i")[0]
            model_id = _unpack(fid, "i")[0]
            width = _unpack(fid, "Q")[0]
            height = _unpack(fid, "Q")[0]
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id} in {path}")
            model, num_params = CAMERA_MODELS[model_id]
            params = _unpack(fid, "d" * num_params)
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def write_cameras_binary(cameras: dict[int, Camera], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras.values():
            if camera.model not in CAMERA_MODEL_NAME_TO_ID:
                raise ValueError(f"Unsupported COLMAP camera model name: {camera.model}")
            model_id = CAMERA_MODEL_NAME_TO_ID[camera.model]
            expected_params = CAMERA_MODELS[model_id][1]
            if len(camera.params) != expected_params:
                raise ValueError(f"{camera.model} expects {expected_params} params, got {len(camera.params)}")
            fid.write(struct.pack("<i", camera.camera_id))
            fid.write(struct.pack("<i", model_id))
            fid.write(struct.pack("<Q", int(camera.width)))
            fid.write(struct.pack("<Q", int(camera.height)))
            fid.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def camera_to_pinhole(camera: Camera) -> Camera:
    if camera.model == "PINHOLE":
        return camera
    if camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = camera.params
        return Camera(camera.camera_id, "PINHOLE", camera.width, camera.height, (focal, focal, cx, cy))
    if camera.model == "SIMPLE_RADIAL":
        focal, cx, cy, _k = camera.params
        return Camera(camera.camera_id, "PINHOLE", camera.width, camera.height, (focal, focal, cx, cy))
    raise ValueError(f"Cannot approximate {camera.model} as PINHOLE")


def read_images_binary(path: str | Path) -> dict[int, ImagePose]:
    path = Path(path)
    images: dict[int, ImagePose] = {}
    with path.open("rb") as fid:
        num_reg_images = _unpack(fid, "Q")[0]
        for _ in range(num_reg_images):
            image_id = _unpack(fid, "i")[0]
            qvec = _unpack(fid, "dddd")
            tvec = _unpack(fid, "ddd")
            camera_id = _unpack(fid, "i")[0]
            name = _read_cstring(fid)
            num_points2d = _unpack(fid, "Q")[0]
            points2d = tuple(_unpack(fid, "ddq") for _ in range(num_points2d))
            images[image_id] = ImagePose(image_id, qvec, tvec, camera_id, name, points2d)
    return images


def read_points3d_binary(path: str | Path) -> dict[int, Point3D]:
    path = Path(path)
    points: dict[int, Point3D] = {}
    with path.open("rb") as fid:
        num_points = _unpack(fid, "Q")[0]
        for _ in range(num_points):
            point3d_id = _unpack(fid, "Q")[0]
            xyz = _unpack(fid, "ddd")
            rgb = _unpack(fid, "BBB")
            error = _unpack(fid, "d")[0]
            track_len = _unpack(fid, "Q")[0]
            track = tuple(_unpack(fid, "ii") for _ in range(track_len))
            points[point3d_id] = Point3D(point3d_id, xyz, rgb, error, track)
    return points


def write_images_binary(images: dict[int, ImagePose], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images.values():
            fid.write(struct.pack("<i", image.image_id))
            fid.write(struct.pack("<dddd", *image.qvec))
            fid.write(struct.pack("<ddd", *image.tvec))
            fid.write(struct.pack("<i", image.camera_id))
            fid.write(image.name.encode("utf-8"))
            fid.write(b"\x00")
            fid.write(struct.pack("<Q", len(image.points2d)))
            for x, y, point3d_id in image.points2d:
                fid.write(struct.pack("<ddq", x, y, int(point3d_id)))


def qvec_normalize(qvec: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(v * v for v in qvec))
    if norm == 0:
        raise ValueError("Quaternion has zero norm")
    return tuple(v / norm for v in qvec)  # type: ignore[return-value]


def qvec2rotmat(qvec: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    qw, qx, qy, qz = qvec_normalize(qvec)
    return (
        (
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qw * qz,
            2 * qx * qz + 2 * qw * qy,
        ),
        (
            2 * qx * qy + 2 * qw * qz,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qw * qx,
        ),
        (
            2 * qx * qz - 2 * qw * qy,
            2 * qy * qz + 2 * qw * qx,
            1 - 2 * qx * qx - 2 * qy * qy,
        ),
    )


def camera_center(
    qvec: tuple[float, float, float, float],
    tvec: tuple[float, float, float],
) -> tuple[float, float, float]:
    # COLMAP stores world-to-camera pose x_cam = R * x_world + t.
    # Camera center in world coordinates is C = -R^T * t.
    rot = qvec2rotmat(qvec)
    return tuple(-sum(rot[row][col] * tvec[row] for row in range(3)) for col in range(3))  # type: ignore[return-value]


def _unpack(fid, fmt: str):
    return struct.unpack("<" + fmt, fid.read(struct.calcsize("<" + fmt)))


def _read_cstring(fid) -> str:
    chars = bytearray()
    while True:
        char = fid.read(1)
        if not char or char == b"\x00":
            break
        chars.extend(char)
    return chars.decode("utf-8", errors="replace")
