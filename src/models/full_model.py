import torch
import torch.nn as nn


class FullModel(nn.Module):
    """
    Unified model:
        - Gaussian generation (ImageEncoder4DG)
        - Semantic extraction (SAM)
        - Semantic alignment to Gaussians
    """

    def __init__(self, semantic_model, gaussian_model, semantic_align):
        super().__init__()

        self.semantic_model = semantic_model
        self.gaussian_model = gaussian_model
        self.semantic_align = semantic_align

    def forward(self, batch):

        images = batch["input_images"]        # [B, T, 3, H, W]
        ts = batch["input_timestamps"]
        K = batch["input_intrinsics"]
        RT = batch["input_extrinsics"]

        # ------------------------------------------------
        # ✅ 1. Gaussian generation
        # ------------------------------------------------
        gs = self.gaussian_model(
            num_input=images.shape[1],
            num_sup=0,
            x=images,
            ts=ts,
            K=K,
            RT=RT,
        )

        # ------------------------------------------------
        # ✅ 2. Semantic (SAM)
        # ------------------------------------------------
        images_sam = images.permute(0, 2, 1, 3, 4)   # [B,3,T,H,W]
        sem_map = self.semantic_model(images_sam)

        # ------------------------------------------------
        # ✅ 3. Align semantic to Gaussians
        # ------------------------------------------------
        gs = self.semantic_align(gs, sem_map)

        return {
            "gs": gs,
            "semantic_map": sem_map,
        }