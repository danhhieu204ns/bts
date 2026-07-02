from __future__ import annotations

from pathlib import Path
import shutil
import struct


SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF,
}


def get_image_size(path: str | Path) -> tuple[int, int]:
    path = Path(path)
    with path.open("rb") as fid:
        header = fid.read(24)

        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)

        if header[:2] == b"\xff\xd8":
            fid.seek(2)
            return _jpeg_size(fid)

    raise ValueError(f"Unsupported or unreadable image format: {path}")


def copy_or_resize_image(src: str | Path, dst: str | Path, size: tuple[int, int]) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if get_image_size(src) == size:
        shutil.copy2(src, dst)
        return

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            f"{src} does not match target size {size}; install Pillow to enable resizing"
        ) from exc

    with Image.open(src) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(size, Image.Resampling.LANCZOS)
        image.save(dst)


def _jpeg_size(fid) -> tuple[int, int]:
    while True:
        byte = fid.read(1)
        while byte and byte != b"\xff":
            byte = fid.read(1)
        if not byte:
            raise ValueError("Could not find JPEG SOF marker")

        marker_byte = fid.read(1)
        while marker_byte == b"\xff":
            marker_byte = fid.read(1)
        if not marker_byte:
            raise ValueError("Unexpected EOF while reading JPEG marker")

        marker = marker_byte[0]
        if marker in (0xD8, 0xD9):
            continue
        if marker in SOF_MARKERS:
            fid.read(3)
            height = struct.unpack(">H", fid.read(2))[0]
            width = struct.unpack(">H", fid.read(2))[0]
            return int(width), int(height)

        segment_length = struct.unpack(">H", fid.read(2))[0]
        fid.seek(segment_length - 2, 1)
