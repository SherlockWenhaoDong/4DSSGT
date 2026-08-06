import torch
import torch.nn as nn
import torch.optim as optim


class Stage1Trainer:
    def __init__(self, model, renderer, device="cuda"):
        self.model = model.to(device)
        self.renderer = renderer
        self.device = device

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-4
        )

        self.l1 = nn.L1Loss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def compute_loss(self, outputs, batch):
        losses = {}

        # ✅ RGB reconstruction
        if "rgb" in outputs:
            losses["rgb"] = self.l1(outputs["rgb"], batch["target_rgb"])

        # ✅ Semantic consistency（关键）
        if "semantic" in outputs and "target_semantic" in batch:
            pred_sem = outputs["semantic"]
            gt_sem = batch["target_semantic"]

            losses["semantic"] = 1 - self.cosine(
                pred_sem, gt_sem
            ).mean()

        total = sum(losses.values())
        return total, losses

    def train_step(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)

        outputs = self.model(
            batch["input_images"],
            batch["input_timestamps"],
            batch["input_intrinsics"],
            batch["input_extrinsics"],
            batch["supervising_timestamps"],
            batch["supervising_intrinsics"],
            batch["supervising_extrinsics"],
            render_mode="RGB",   # or "SEM"
        )

        loss, loss_dict = self.compute_loss(outputs, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_dict

    def train(self, dataloader, epochs=10):
        for epoch in range(epochs):
            for batch in dataloader:
                loss, logs = self.train_step(batch)

                print(f"[Stage1] Loss: {loss:.4f}", logs)