import torch
import torch.nn as nn
import torch.optim as optim


class GaussianRefinementTrainerV2:
    def __init__(self, renderer, gs_params, device="cuda", lr=5e-3):
        self.renderer = renderer
        self.device = device

        self.gs_params = {
            k: torch.nn.Parameter(v.to(device))
            for k, v in gs_params.items()
        }

        self.optimizer = optim.Adam(self.gs_params.values(), lr=lr)
        self.l1 = nn.L1Loss()

    def prune(self, outputs, opacity_th=0.01):
        with torch.no_grad():
            opacity = self.gs_params["opacity"].squeeze(-1)  # (N,)

            if "mask" in outputs:
                alpha = outputs["mask"].mean(dim=(0,2,3))  # approx importance
            else:
                alpha = torch.ones_like(opacity)

            importance = opacity * alpha

            mask = importance > opacity_th

            for k in self.gs_params:
                self.gs_params[k].data = self.gs_params[k].data[mask]

    def densify_by_gradient(self, grad_th=2e-4, noise_scale=1e-3):
        with torch.no_grad():
            grad = self.gs_params["xyz"].grad
            if grad is None:
                return

            grad_norm = torch.norm(grad, dim=-1)
            mask = grad_norm > grad_th

            if mask.sum() == 0:
                return

            new_params = {}

            for k, v in self.gs_params.items():
                selected = v[mask]

                if k == "xyz":
                    scale = self.gs_params["scaling"][mask]
                    noise = torch.randn_like(selected) * scale * noise_scale
                    new_v = torch.cat([v, selected + noise], dim=0)

                elif k == "scaling":
                    new_v = torch.cat([v, selected * 0.8], dim=0)

                else:
                    new_v = torch.cat([v, selected], dim=0)

                new_params[k] = new_v

            self.gs_params = {
                k: torch.nn.Parameter(v)
                for k, v in new_params.items()
            }

            self.optimizer = optim.Adam(self.gs_params.values(), lr=5e-4)

    def densify_by_semantic(self, semantic, grad_th=0.2, noise_scale=1e-3):
        """
        semantic: (N, C)
        """

        with torch.no_grad():
            idx = torch.randint(0, semantic.shape[0], (semantic.shape[0],), device=semantic.device)
            neighbor = semantic[idx]

            grad = torch.norm(semantic - neighbor, dim=-1)

            mask = grad > grad_th

            if mask.sum() == 0:
                return

            new_params = {}

            for k, v in self.gs_params.items():
                selected = v[mask]

                if k == "xyz":
                    noise = torch.randn_like(selected) * noise_scale
                    new_v = torch.cat([v, selected + noise], dim=0)

                elif k == "scaling":
                    new_v = torch.cat([v, selected * 0.6], dim=0)

                else:
                    new_v = torch.cat([v, selected], dim=0)

                new_params[k] = new_v

            self.gs_params = {
                k: torch.nn.Parameter(v)
                for k, v in new_params.items()
            }

            self.optimizer = optim.Adam(self.gs_params.values(), lr=5e-4)

    # ------------------------------------------------
    # ✅ Train step
    # ------------------------------------------------
    def train_step(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)

        outputs = self.renderer(
            self.gs_params,
            batch["supervising_timestamps"],
            batch["supervising_intrinsics"],
            batch["supervising_extrinsics"],
            render_mode="RGB",
        )

        loss = self.l1(outputs["rgb"], batch["target_rgb"])

        self.optimizer.zero_grad()
        loss.backward()

        self.densify_by_gradient()

        self.optimizer.step()

        return loss.item(), outputs

    # ------------------------------------------------
    # ✅ Training loop
    # ------------------------------------------------
    def train(
        self,
        dataloader,
        iterations=10000,
        prune_interval=500,
        sem_interval=800,
    ):
        step = 0

        while step < iterations:
            for batch in dataloader:

                loss, outputs = self.train_step(batch)

                if step % 50 == 0:
                    print(f"[Stage2+] Step {step} | Loss {loss:.4f}")

                if step > 0 and step % prune_interval == 0:
                    print(">>> Smart Pruning...")
                    self.prune(outputs)

                # ✅ Semantic densify
                if step > 0 and step % sem_interval == 0:
                    print(">>> Semantic-aware Densify...")
                    if "semantic" in self.gs_params:
                        self.densify_by_semantic(
                            self.gs_params["semantic"].detach()
                        )

                step += 1
                if step >= iterations:
                    break
