from __future__ import annotations

import torch
from gsplat import rasterization


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    num_gaussians = 10
    means = torch.randn(num_gaussians, 3, device="cuda")
    means[:, 2] += 4
    quats = torch.zeros(num_gaussians, 4, device="cuda")
    quats[:, 0] = 1
    scales = torch.full((num_gaussians, 3), 0.05, device="cuda")
    opacities = torch.full((num_gaussians,), 0.8, device="cuda")
    colors = torch.rand(num_gaussians, 3, device="cuda")
    viewmats = torch.eye(4, device="cuda")[None]
    intrinsics = torch.tensor(
        [[[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]],
        device="cuda",
    )

    image, alpha, _meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        64,
        64,
        backgrounds=torch.ones(1, 3, device="cuda"),
    )
    print(f"image={tuple(image.shape)} alpha={tuple(alpha.shape)} min={image.min().item():.6f} max={image.max().item():.6f}")


if __name__ == "__main__":
    main()
