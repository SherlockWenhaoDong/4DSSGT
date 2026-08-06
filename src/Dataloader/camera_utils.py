import torch


def build_default_intrinsics(H, W, device="cpu"):
    """
    Simple pinhole camera:
        fx = fy = focal
        cx = W/2
        cy = H/2
    """

    focal = max(H, W)  # reasonable default

    K = torch.tensor([
        [focal, 0.0,   W / 2],
        [0.0,   focal, H / 2],
        [0.0,   0.0,   1.0]
    ], dtype=torch.float32).to(device)

    return K


def build_fixed_extrinsics(device="cpu"):
    """
    Identity camera pose (no motion)
    """

    RT = torch.eye(4, dtype=torch.float32).to(device)
    return RT


def build_timestamp(idx, fps=30):
    """
    Convert frame index → time (seconds)
    """
    return torch.tensor([idx / fps], dtype=torch.float32)