from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .colmap_io import qvec2rotmat
from .dataset import Scene, TestPose, find_scenes, read_test_poses


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BTS test_poses.csv with trained official 3DGS models.")
    parser.add_argument("--data-root", nargs="+", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument("--antialiasing", action="store_true", help="Enable 3DGS rasterizer antialiasing at render time.")
    parser.add_argument("--scaling-modifier", type=float, default=1.0, help="Gaussian scale multiplier for render-time tuning.")
    parser.add_argument("--zip", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    gs_root = repo_root / "external" / "gaussian-splatting"
    if not gs_root.is_dir():
        raise FileNotFoundError(f"official 3DGS repo not found: {gs_root}")
    sys.path.insert(0, str(gs_root))

    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
        separate_sh = True
    except Exception:
        separate_sh = False

    scenes = find_scenes(args.data_root)
    if args.scene:
        scene_key = args.scene.lower()
        scenes = [scene for scene in scenes if scene.name.lower() == scene_key]
        if not scenes:
            raise ValueError(f"Scene not found in data roots: {args.scene}")

    model_root = Path(args.model_root)
    out_dir = Path(args.out_dir)
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=args.antialiasing,
    )
    bg_value = 1.0 if args.background == "white" else 0.0
    background = torch.tensor([bg_value, bg_value, bg_value], dtype=torch.float32, device="cuda")

    for scene in scenes:
        model_path = _resolve_model_path(model_root, scene)
        iteration = _resolve_iteration(model_path, args.iteration)
        ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply_path.is_file():
            raise FileNotFoundError(f"Trained point cloud not found: {ply_path}")

        print(f"Rendering {scene.name} from {ply_path}")
        gaussians = GaussianModel(args.sh_degree)
        gaussians.load_ply(str(ply_path))

        scene_out_dir = out_dir / scene.name
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        test_poses = read_test_poses(scene.test_poses_csv)
        for idx, pose in enumerate(tqdm(test_poses, desc=scene.name)):
            camera = _camera_from_pose(pose, idx, focal2fov, getProjectionMatrix, getWorld2View2)
            rendering = render(
                camera,
                gaussians,
                pipe,
                background,
                scaling_modifier=args.scaling_modifier,
                separate_sh=separate_sh,
                use_trained_exp=False,
            )["render"]
            _save_render(rendering, scene_out_dir / pose.image_name)

        del gaussians
        torch.cuda.empty_cache()

    if args.zip:
        _zip_submission(out_dir, Path(args.zip))


def _resolve_model_path(model_root: Path, scene: Scene) -> Path:
    scene_model = model_root / scene.name
    if (scene_model / "point_cloud").is_dir():
        return scene_model
    if (model_root / "point_cloud").is_dir():
        return model_root
    raise FileNotFoundError(
        f"No 3DGS model found for {scene.name}. Expected {scene_model} or direct model root {model_root}"
    )


def _resolve_iteration(model_path: Path, iteration: int) -> int:
    if iteration >= 0:
        return iteration

    point_cloud_dir = model_path / "point_cloud"
    iterations: list[int] = []
    for path in point_cloud_dir.glob("iteration_*"):
        if path.is_dir():
            try:
                iterations.append(int(path.name.removeprefix("iteration_")))
            except ValueError:
                pass
    if not iterations:
        raise FileNotFoundError(f"No iteration_* directories found under {point_cloud_dir}")
    return max(iterations)


def _camera_from_pose(pose: TestPose, uid: int, focal2fov, getProjectionMatrix, getWorld2View2):
    r_world_to_cam = np.array(qvec2rotmat(pose.qvec), dtype=np.float32)
    r_for_3dgs = r_world_to_cam.transpose()
    tvec = np.array(pose.tvec, dtype=np.float32)
    fovx = focal2fov(pose.fx, pose.width)
    fovy = focal2fov(pose.fy, pose.height)
    world_view_transform = torch.tensor(getWorld2View2(r_for_3dgs, tvec)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=fovx,
        fovY=fovy,
    ).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return SimpleNamespace(
        uid=uid,
        colmap_id=uid,
        R=r_for_3dgs,
        T=tvec,
        FoVx=fovx,
        FoVy=fovy,
        image_name=pose.image_name,
        image_width=pose.width,
        image_height=pose.height,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=world_view_transform.inverse()[3, :3],
    )


def _save_render(rendering: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = rendering.detach().clamp(0.0, 1.0).mul(255.0).add(0.5).byte()
    array = image.permute(1, 2, 0).cpu().numpy()
    save_kwargs = {"quality": 95} if path.suffix.lower() in {".jpg", ".jpeg"} else {}
    Image.fromarray(array, mode="RGB").save(path, **save_kwargs)


def _zip_submission(out_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_dir).as_posix())
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
