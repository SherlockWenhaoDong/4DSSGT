import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSemanticAlign(nn.Module):

    def forward(self, gs_params, sem_map):

        xyz = gs_params["xyz"]

        sem = sem_map

        coords = xyz[..., :2]

        x = coords[..., 0]
        y = coords[..., 1]

        x = x / (x.abs().max() + 1e-6)
        y = y / (y.abs().max() + 1e-6)

        coords = torch.stack([x, y], dim=-1)

        coords = coords.unsqueeze(2)


        sampled = F.grid_sample(
            sem,
            coords,
            mode="bilinear",
            align_corners=True,
        )

        sampled = sampled.squeeze(-1).permute(0, 2, 1)

        new_gs_params = dict(gs_params)
        new_gs_params["semantic"] = sampled

        return new_gs_params

