import torch
import torch.nn as nn
import torch.optim as optim


class End2EndTrainer:

    def __init__(
        self,
        model,
        renderer,
        device="cuda",
        stage1_iters=5000,
        stage2_iters=5000,
        lr_stage1=1e-3,
        lr_stage2=5e-3,
    ):

        self.model = model.to(device)
        self.renderer = renderer
        self.device = device

        self.stage1_iters = stage1_iters
        self.stage2_iters = stage2_iters
        self.total_iters = stage1_iters + stage2_iters

        self.stage = 1

        # ✅ losses
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

        # ✅ optimizer
        self.opt_stage1 = optim.AdamW(self.model.parameters(), lr=lr_stage1)
        self.opt_stage2 = None  # initialized in stage2

        self.gs_params = None

    # ------------------------------------------------
    # ✅ Metrics
    # ------------------------------------------------
    def compute_psnr(self, pred, gt):
        mse = torch.mean((pred - gt) ** 2)
        return -10 * torch.log10(mse + 1e-8)

    def compute_miou(self, pred, gt):

        pred = torch.argmax(pred, dim=1)   # [B,H,W]
        gt = gt.squeeze(1)                 # [B,H,W]

        num_classes = pred.max() + 1

        ious = []
        for cls in range(num_classes):

            inter = ((pred == cls) & (gt == cls)).sum()
            union = ((pred == cls) | (gt == cls)).sum()

            if union == 0:
                continue

            ious.append(inter.float() / (union + 1e-6))

        if len(ious) == 0:
            return torch.tensor(0.0, device=pred.device)

        return torch.mean(torch.stack(ious))

    def compute_pix_acc(self, pred, gt):

        pred = torch.argmax(
            pred,
            dim=1,
        )

        correct = (
                pred == gt
        ).float()

        return correct.mean()

    # ------------------------------------------------
    # ✅ Stage2 init
    # ------------------------------------------------
    def switch_to_stage2(self, batch):

        print("\n===== SWITCH TO STAGE 2 =====\n")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(batch)

        self.gs_params = {
            k: torch.nn.Parameter(v.detach().clone().to(self.device))
            for k, v in outputs["gs"].items()
        }

        self.opt_stage2 = optim.Adam(self.gs_params.values(), lr=5e-3)

        self.stage = 2

    # ------------------------------------------------
    # ✅ Stage1
    # ------------------------------------------------
    def train_step_stage1(self, batch):

        self.opt_stage1.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(batch)

            render_out = self.renderer(
                outputs["gs"],
                batch["input_timestamps"],
                batch["input_intrinsics"],
                batch["input_extrinsics"],
            )
        # ✅ RGB loss
        loss = self.l1(render_out["rgb"], batch["target_rgb"])

        # ✅ Semantic loss
        if "semantic" in render_out:
            pred = render_out["semantic"]

            B, T, C, H, W = pred.shape

            pred = pred.reshape(B * T, C, H, W).contiguous()

            gt = batch["semantic_mask"]
            gt = gt.unsqueeze(1).repeat(1, T, 1, 1)
            gt = gt.reshape(B * T, H, W).contiguous()

            gt = torch.clamp(gt, 0, C - 1)

            loss_sem = self.ce(pred, gt.long())
            loss += loss_sem

        else:
            loss_sem = None

        loss.backward()
        self.opt_stage1.step()

        # ✅ metrics
        psnr = self.compute_psnr(render_out["rgb"], batch["target_rgb"])

        acc = None

        if loss_sem is not None:
            acc = self.compute_pix_acc(
                pred,
                gt
            )

        return loss.item(), psnr.item(), (acc.item() if acc is not None else None), render_out

    def prune(
            self,
            opacity_th=0.01,
            scale_th=1e-4,
            entropy_th=0.05,
            min_keep=10000,
    ):
        with torch.no_grad():

            B = self.gs_params["xyz"].shape[0]

            keep_mask = None

            # --------------------------------
            # opacity
            # --------------------------------
            opacity = self.gs_params["opacity"].squeeze(-1)

            keep_mask = opacity > opacity_th

            # --------------------------------
            # scale
            # --------------------------------
            if "scaling" in self.gs_params:
                scale = self.gs_params["scaling"]

                scale_keep = (
                        scale.mean(-1) > scale_th
                )

                keep_mask = keep_mask & scale_keep

            # --------------------------------
            # semantic entropy
            # --------------------------------
            if "semantic" in self.gs_params:
                prob = torch.softmax(
                    self.gs_params["semantic"],
                    dim=-1,
                )

                entropy = -(
                        prob *
                        torch.log(prob + 1e-6)
                ).sum(-1)

                keep_mask = (
                        keep_mask &
                        (entropy < entropy_th)
                )

            # --------------------------------
            # prevent deleting all
            # --------------------------------
            new_params = {}

            for k, v in self.gs_params.items():

                kept_batches = []

                for b in range(B):

                    cur_mask = keep_mask[b]

                    if cur_mask.sum() < min_keep:
                        score = opacity[b]

                        _, idx = torch.topk(
                            score,
                            k=min(
                                min_keep,
                                score.shape[0],
                            ),
                        )

                        cur_mask = torch.zeros_like(
                            cur_mask,
                            dtype=torch.bool,
                        )

                        cur_mask[idx] = True

                    kept = v[b][cur_mask]

                    kept_batches.append(kept)

                min_n = min(
                    x.shape[0]
                    for x in kept_batches
                )

                kept_batches = [
                    x[:min_n]
                    for x in kept_batches
                ]

                new_params[k] = torch.stack(
                    kept_batches,
                    dim=0,
                )

            self.gs_params = {
                k: nn.Parameter(v)
                for k, v in new_params.items()
            }

            self.opt_stage2 = optim.Adam(
                self.gs_params.values(),
                lr=5e-4,
            )

            print(
                "Pruned Gaussian:",
                self.gs_params["xyz"].shape
            )

    def densify(
            self,
            grad_th=2e-4,
            sem_th=0.2,
            max_gaussians=1000000,
    ):

        with torch.no_grad():

            cur_n = self.gs_params["xyz"].shape[1]

            if cur_n > max_gaussians:
                print(
                    f"Skip densify. "
                    f"Gaussian={cur_n}"
                )
                return

            grad = self.gs_params["xyz"].grad

            if grad is None:
                return

            grad_norm = torch.norm(
                grad,
                dim=-1,
            )

            if "semantic" in self.gs_params:

                sem = torch.softmax(
                    self.gs_params["semantic"],
                    dim=-1,
                )

                sem_mean = sem.mean(
                    dim=1,
                    keepdim=True,
                )

                sem_diff = torch.norm(
                    sem - sem_mean,
                    dim=-1,
                )

            else:

                sem_diff = torch.zeros_like(
                    grad_norm
                )

            clone_mask = (
                    (grad_norm > grad_th)
                    |
                    (sem_diff > sem_th)
            )

            new_params = {}

            for k, v in self.gs_params.items():

                clone_list = []

                B = v.shape[0]

                for b in range(B):

                    selected = v[b][clone_mask[b]]

                    if selected.shape[0] == 0:
                        selected = v[b][
                            torch.randint(
                                0,
                                v.shape[1],
                                (128,),
                                device=v.device,
                            )
                        ]

                    max_clone = min(
                        512,
                        int(v.shape[1] * 0.005)
                    )

                    selected = selected[:max_clone]

                    if k == "xyz":

                        selected = (
                                selected
                                + torch.randn_like(selected)
                                * 1e-3
                        )

                    elif k == "scaling":

                        selected = torch.clamp(
                            selected * 0.5,
                            min=1e-4,
                        )


                    elif k == "opacity":

                        selected = selected * 0.5

                        v[b][clone_mask[b]][:selected.shape[0]] *= 0.5

                    clone_list.append(
                        selected
                    )

                clone_list = torch.stack(
                    clone_list,
                    dim=0,
                )

                new_params[k] = torch.cat(
                    [
                        v,
                        clone_list,
                    ],
                    dim=1,
                )

            self.gs_params = {
                k: nn.Parameter(v)
                for k, v in new_params.items()
            }

            self.opt_stage2 = optim.Adam(
                self.gs_params.values(),
                lr=5e-4,
            )

            print(
                "Densify ->",
                self.gs_params["xyz"].shape
            )

    # ------------------------------------------------
    # ✅ Stage2
    # ------------------------------------------------
    def train_step_stage2(self, batch, step):

        self.opt_stage2.zero_grad()

        render_out = self.renderer(
            self.gs_params,
            batch["input_timestamps"],
            batch["input_intrinsics"],
            batch["input_extrinsics"],
        )

        # ✅ RGB loss
        loss = self.l1(render_out["rgb"], batch["target_rgb"])

        # ✅ Semantic loss
        if "semantic" in render_out:
            pred = render_out["semantic"]

            B, T, C, H, W = pred.shape

            pred = pred.reshape(B * T, C, H, W).contiguous()

            gt = batch["semantic_mask"]
            gt = gt.unsqueeze(1).repeat(1, T, 1, 1)
            gt = gt.reshape(B * T, H, W).contiguous()

            gt = torch.clamp(gt, 0, C - 1)

            loss_sem = self.ce(pred, gt.long())
            loss += loss_sem
        else:
            loss_sem = None

        loss.backward()

        if step % 1000 == 0 and step > 3000:
            self.densify()

        self.opt_stage2.step()

        with torch.no_grad():

            self.gs_params["scaling"].clamp_(
                min=1e-4,
                max=1.0,
            )

            self.gs_params["opacity"].clamp_(
                min=1e-4,
                max=1.0,
            )

            self.gs_params["cov_t"].clamp_(
                min=1e-4,
                max=100.0,
            )

        # ✅ prune
        if step % 5000 == 0:
            print(">>> Pruning")
            self.prune()

        psnr = self.compute_psnr(render_out["rgb"], batch["target_rgb"])
        acc = None

        if loss_sem is not None:
            acc = self.compute_pix_acc(
                pred,
                gt
            )

        return loss.item(), psnr.item(), (acc.item() if acc is not None else None), render_out

    # ------------------------------------------------
    # ✅ TRAIN LOOP
    # ------------------------------------------------
    def train(self, dataloader):

        step = 0

        while step < self.total_iters:

            for batch in dataloader:

                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.stage == 1:

                    loss, psnr, miou = self.train_step_stage1(batch)

                    if step == self.stage1_iters:
                        self.switch_to_stage2(batch)

                else:
                    loss, psnr = self.train_step_stage2(batch, step)
                    miou = None

                if step % 50 == 0:
                    print(
                        f"[E2E] Step {step} | Stage {self.stage} "
                        f"| Loss {loss:.4f} | PSNR {psnr:.2f} "
                        f"| mIoU {miou if miou else 'N/A'}"
                    )

                step += 1
                if step >= self.total_iters:
                    break