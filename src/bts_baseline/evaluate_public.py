from __future__ import annotations

import argparse
import csv
from math import log10
from pathlib import Path
from typing import Any

from .dataset import find_scenes, read_test_poses


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate public predictions with available ground truth.")
    parser.add_argument("--data-root", nargs="+", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--lpips", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--psnr-max", nargs="+", type=float, default=[30.0, 35.0, 40.0])
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Install requirements first: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt") from exc

    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        structural_similarity = None

    lpips_bundle = load_lpips(args.lpips, args.lpips_net, args.device)
    pred_dir = Path(args.pred_dir)
    scenes = [scene for scene in find_scenes(args.data_root) if scene.test_images_dir.is_dir()]
    if not scenes:
        raise SystemExit("No public scenes with test/images found.")

    image_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene in scenes:
        scene_psnr: list[float] = []
        scene_ssim: list[float] = []
        scene_lpips: list[float] = []
        for pose in read_test_poses(scene.test_poses_csv):
            pred_path = pred_dir / scene.name / pose.image_name
            gt_path = scene.test_images_dir / pose.image_name
            if not pred_path.is_file() or not gt_path.is_file():
                continue

            pred = _load_rgb_array(Image, np, pred_path)
            gt = _load_rgb_array(Image, np, gt_path)
            if pred.shape != gt.shape:
                raise RuntimeError(f"Shape mismatch for {scene.name}/{pose.image_name}: {pred.shape} vs {gt.shape}")

            psnr_value = psnr(np, pred, gt)
            scene_psnr.append(psnr_value)

            ssim_value = None
            if structural_similarity is not None:
                ssim_value = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
                scene_ssim.append(ssim_value)

            lpips_value = None
            if lpips_bundle is not None:
                lpips_value = compute_lpips(lpips_bundle, pred, gt)
                scene_lpips.append(lpips_value)

            image_rows.append(
                {
                    "scene": scene.name,
                    "image_name": pose.image_name,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                }
            )

        summary = summarize(scene.name, scene_psnr, scene_ssim, scene_lpips, args.psnr_max)
        scene_summaries.append(summary)
        print(format_summary(summary, args.psnr_max))

    overall = average_summaries("OVERALL(scene-avg)", scene_summaries, args.psnr_max)
    print(format_summary(overall, args.psnr_max))
    if lpips_bundle is None:
        print("LPIPS skipped; local final_score is unavailable until torch + lpips are installed/enabled.")

    if args.output_csv:
        write_image_metrics(Path(args.output_csv), image_rows)
        print(f"Wrote image metrics: {args.output_csv}")


def _load_rgb_array(Image, np, path: Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def psnr(np, pred, gt) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * log10(255.0) - 10.0 * log10(mse)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_lpips(mode: str, net: str, device_arg: str):
    if mode == "off":
        return None
    try:
        import lpips
        import torch
    except ImportError as exc:
        if mode == "on":
            raise SystemExit(
                "LPIPS requested but not installed. Install optional deps with: "
                ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-lpips.txt"
            ) from exc
        return None

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested for LPIPS, but torch.cuda.is_available() is false.")

    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    print(f"LPIPS enabled: net={net}, device={device}")
    return {"torch": torch, "model": model, "device": device}


def compute_lpips(bundle: dict[str, Any], pred, gt) -> float:
    torch = bundle["torch"]
    model = bundle["model"]
    device = bundle["device"]

    pred_tensor = image_array_to_lpips_tensor(torch, pred, device)
    gt_tensor = image_array_to_lpips_tensor(torch, gt, device)
    with torch.no_grad():
        value = model(pred_tensor, gt_tensor)
    return float(value.item())


def image_array_to_lpips_tensor(torch, array, device: str):
    tensor = torch.from_numpy(array / 127.5 - 1.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=torch.float32)


def summarize(
    scene_name: str,
    psnr_values: list[float],
    ssim_values: list[float],
    lpips_values: list[float],
    psnr_max_values: list[float],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scene": scene_name,
        "n": len(psnr_values),
        "psnr": _mean(psnr_values),
        "ssim": _mean(ssim_values),
        "lpips": _mean(lpips_values),
    }
    for psnr_max in psnr_max_values:
        summary[score_key(psnr_max)] = final_score(summary["psnr"], summary["ssim"], summary["lpips"], psnr_max)
    return summary


def average_summaries(
    scene_name: str,
    summaries: list[dict[str, Any]],
    psnr_max_values: list[float],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scene": scene_name,
        "n": sum(int(item["n"]) for item in summaries),
        "psnr": _mean([item["psnr"] for item in summaries]),
        "ssim": _mean([item["ssim"] for item in summaries]),
        "lpips": _mean([item["lpips"] for item in summaries]),
    }
    for psnr_max in psnr_max_values:
        key = score_key(psnr_max)
        summary[key] = _mean([item[key] for item in summaries])
    return summary


def final_score(psnr_value: float, ssim_value: float, lpips_value: float, psnr_max: float) -> float:
    if any(value != value for value in (psnr_value, ssim_value, lpips_value)):
        return float("nan")
    psnr_norm = min(max(psnr_value / psnr_max, 0.0), 1.0)
    return 0.4 * (1.0 - lpips_value) + 0.3 * ssim_value + 0.3 * psnr_norm


def score_key(psnr_max: float) -> str:
    return f"score_psnrmax_{psnr_max:g}"


def format_summary(summary: dict[str, Any], psnr_max_values: list[float]) -> str:
    parts = [
        f"{summary['scene']}: n={summary['n']}",
        f"PSNR={summary['psnr']:.4f}",
        f"SSIM={summary['ssim']:.4f}",
    ]
    if summary["lpips"] == summary["lpips"]:
        parts.append(f"LPIPS={summary['lpips']:.4f}")
    for psnr_max in psnr_max_values:
        key = score_key(psnr_max)
        value = summary[key]
        if value == value:
            parts.append(f"Score@PSNRmax{psnr_max:g}={value:.4f}")
    return " | ".join(parts)


def write_image_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene", "image_name", "psnr", "ssim", "lpips"]
    with path.open("w", encoding="utf-8", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
