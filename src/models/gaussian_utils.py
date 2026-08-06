import torch

def prune_gaussians(gs, opacity_th=0.01):
    mask = gs["opacity"] > opacity_th

    for k in gs:
        gs[k] = gs[k][mask]

    return gs


def densify_gaussians(gs, factor=2):
    new_gs = {}

    for k in gs:
        if k in ["xyz", "t"]:
            noise = torch.randn_like(gs[k]) * 0.001
            new_gs[k] = torch.cat([gs[k], gs[k] + noise], dim=1)
        else:
            new_gs[k] = torch.cat([gs[k], gs[k]], dim=1)

    return new_gs