# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Mapping

import torch
from torch import nn

from src.gsplat.gsplat.rendering import rasterization, rasterization_rgb_semantic

from ..acceleration.checkpoint import auto_grad_checkpoint
from ..easyvolcap.utils.math_utils import affine_inverse
from ..registry import RENDERER
from .gaussian_renderer import GaussianRenderer


@torch.jit.script
def compute_marginal_t(t: torch.Tensor, mu_t: torch.Tensor, cov_t: torch.Tensor):
    return torch.exp(-0.5 * (t - mu_t) ** 2 / cov_t)


class SingleImageRasterization(nn.Module):
    def __init__(self, height: int, width: int, znear: float, zfar: float):
        super().__init__()
        self.height = height
        self.width = width
        self.znear = znear
        self.zfar = zfar

    def forward(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,      # [..., N, 3 + S]
        viewmats,
        Ks,
        height=0,
        width=0,
        render_mode="RGB+ED",
    ):
        C_cam = viewmats.shape[-3]
        H = height if height else self.height
        W = width if width else self.width
        D = colors.shape[-1]  # 3 + semantic


        backgrounds = torch.zeros(
            (C_cam, H, W, D),
            device=colors.device,
            dtype=colors.dtype,
        )

        out_rgb, out_semantic, out_alpha, out_dpt, meta = rasterization_rgb_semantic(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            backgrounds=backgrounds,
            width=W,
            height=H,
            near_plane=self.znear,
            far_plane=self.zfar,
            render_mode=render_mode,
        )

        out_rgb = out_rgb.contiguous()
        out_alpha = out_alpha.contiguous()
        out_dpt = out_dpt.contiguous()

        if out_semantic is not None:
            out_semantic = out_semantic.contiguous()

        return out_rgb, out_semantic, out_alpha, out_dpt, meta


class SingleBatchRasterization(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        znear: float,
        zfar: float,
        marginal_th: float = 0.05,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.znear = znear
        self.zfar = zfar
        self.single_image_rasterizer = SingleImageRasterization(
            height, width, znear, zfar
        )
        self.marginal_th = marginal_th

    def forward(
        self,
        xyz,
        rgb,
        scale,
        rotation,
        opacity,
        t,
        cov_t,
        ms3,
        w2cs,
        Ks,
        ts,
        semantic=None,
        height=0,
        width=0,
        render_mode="RGB",
    ):
        """
        Unified rendering:
        RGB and semantic are concatenated and splatted together.
        """

        # B, T, _, _ = w2cs.shape

        B = xyz.shape[0]
        T = ts.shape[1]

        out_rgbs = []
        out_sems = []
        out_masks = []
        out_dpts = []

        for b in range(xyz.shape[0]):

            rgb_list = []
            sem_list = []
            mask_list = []
            dpt_list = []

            assert xyz.shape[0] > 0
            assert ms3.shape[0] > 0
            assert t.shape[0] > 0
            assert ts.shape[0] > 0


            for i in range(T):
                means = xyz[b].float() + (ms3[b] * (ts[b, i] - t[b])).float()
                quats = rotation[b].float()
                scales = scale[b].float()

                opacities = (
                    opacity[b].float()
                    * compute_marginal_t(ts[b, i], t[b], cov_t[b]).float()
                )[..., 0]

                # ✅ ---------- KEY: concatenate features ----------
                if semantic is not None:
                    colors = torch.cat(
                        [
                            rgb[b, ..., :3],        # RGB
                            semantic[b],            # semantic logits
                        ],
                        dim=-1,
                    ).float()
                else:
                    colors = rgb[b, ..., :3].float()
                # ------------------------------------------------

                viewmats = w2cs[b, i : i + 1].float()
                ixts = Ks[b, i : i + 1].float()

                out_img, out_sem, out_alpha, out_dpt, _ = self.single_image_rasterizer(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors,
                    viewmats,
                    ixts,
                    height,
                    width,
                    "RGB+ED",
                )

                rgb = out_img
                sem = out_sem

                rgb = rgb.permute(0, 3, 1, 2).contiguous()
                rgb = (2 * rgb - 1).clamp(-1, 1)

                out_alpha = out_alpha.permute(0, 3, 1, 2).contiguous()
                out_dpt = out_dpt.permute(0, 3, 1, 2).contiguous()

                if sem is not None:
                    sem = sem.permute(0, 3, 1, 2).contiguous()

                rgb_list.append(rgb)
                mask_list.append(out_alpha)
                dpt_list.append(out_dpt)
                if sem is not None:
                    sem_list.append(sem)

            rgb = torch.cat(rgb_list)
            mask = torch.cat(mask_list)
            depth = torch.cat(dpt_list)

            if semantic is not None and len(sem_list) > 0:
                sem = torch.cat(sem_list)
            else:
                sem = None

            out_rgbs.append(rgb)
            out_masks.append(mask)
            out_dpts.append(depth)
            out_sems.append(sem)

        rgba = torch.stack(out_rgbs).contiguous()
        mask = torch.stack(out_masks).contiguous()
        depth = torch.stack(out_dpts).contiguous()

        if semantic is not None:
            sem = torch.stack(out_sems).contiguous()
        else:
            sem = None

        return rgba, mask, depth, sem


@RENDERER.register_module()
class GaussianRenderer4D(GaussianRenderer):
    def __init__(
        self,
        height=512,
        width=512,
        znear=0.01,
        zfar=500,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.znear = znear
        self.zfar = zfar

        self.single_batch_rasterizer = SingleBatchRasterization(
            height, width, znear, zfar
        )

    def __call__(
        self,
        gs_params,
        ts,
        Ks,
        RTs,
        height=0,
        width=0,
        **kwargs,
    ):
        """
        One-pass rendering of RGB + semantic
        """

        xyz = gs_params["xyz"]
        rgb = gs_params["feature"]
        scale = gs_params["scaling"]
        rotation = gs_params["rotation"]
        opacity = gs_params["opacity"]

        t = gs_params["t"]
        cov_t = gs_params["cov_t"]
        ms3 = gs_params["ms3"]

        semantic = gs_params.get("semantic", None)

        w2cs = affine_inverse(RTs)

        imgs, masks, depths, sems = [], [], [], []

        assert xyz.shape[0] == ts.shape[0], (
            f"Batch mismatch: "
            f"xyz={xyz.shape}, "
            f"ts={ts.shape}"
        )

        B = xyz.shape[0]

        for b in range(B):

            img, mask, depth, sem = self.single_batch_rasterizer(
                xyz[b : b + 1],
                rgb[b : b + 1],
                scale[b : b + 1],
                rotation[b : b + 1],
                opacity[b : b + 1],
                t[b : b + 1],
                cov_t[b : b + 1],
                ms3[b : b + 1],
                w2cs[b : b + 1],
                Ks[b : b + 1],
                ts[b : b + 1],
                semantic=semantic[b : b + 1] if semantic is not None else None,
                height=height,
                width=width,
            )

            imgs.append(img)
            masks.append(mask)
            depths.append(depth)

            if sem is not None:
                sems.append(sem)

        img = torch.cat(imgs).contiguous()
        mask = torch.cat(masks).contiguous()
        depth = torch.cat(depths).contiguous()


        output = {
            "rgb": img,
            "mask": mask,
            "depth": depth,
        }

        if semantic is not None:
            semantic_map = torch.cat(sems).contiguous()
            output["semantic"] = semantic_map

        return output