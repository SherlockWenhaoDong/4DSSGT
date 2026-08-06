import yaml
import torch

from itertools import cycle

from src.models.registry import MODELS, RENDERER, build_module

import src.models.encoders.Gaussian_encoder
import src.models.encoders.Semantic_encoder
import src.models.renderers.gaussian_renderer_4D

from src.models.full_model import FullModel
from src.models.aligner import GaussianSemanticAlign
from src.trainer.end2end import End2EndTrainer

from src.Dataloader.VideoDataset import build_dataloaders
import os
import torchvision

import torch.multiprocessing as mp
from tqdm.auto import tqdm
import time
mp.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import os

os.environ["PATH"] = (
    "/home/wdong/miniconda3/envs/4dgt/bin:"
    + os.environ["PATH"]
)


CONDA_BIN = "/home/wdong/miniconda3/envs/4dgt/bin"

os.environ["PATH"] = CONDA_BIN + ":" + os.environ.get("PATH", "")

os.environ["CC"] = f"{CONDA_BIN}/x86_64-conda-linux-gnu-gcc"
os.environ["CXX"] = f"{CONDA_BIN}/x86_64-conda-linux-gnu-g++"
os.environ["CUDAHOSTCXX"] = os.environ["CXX"]

COLOR_MAP = torch.tensor([
    [255,128,0],

    [255,0,0],
    [0,255,0],
    [0,0,255],

    [255,255,0],
    [255,0,255],

    [0,255,255],
    [128,0,0],

    [0,128,0],
    [0,0,128],

    [128,128,0],
    [128,0,128],
], dtype=torch.uint8)
# ------------------------------------------------
# Load config
# ------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------
# Build model and renderer
# ------------------------------------------------
def build_all(cfg, device):

    semantic_model = build_module(
        cfg["model_semantic"],
        MODELS
    ).to(device)

    gaussian_model = build_module(
        cfg["model"],
        MODELS
    ).to(device)

    semantic_align = GaussianSemanticAlign().to(device)

    renderer = build_module(
        cfg["renderer"],
        RENDERER
    )

    full_model = FullModel(
        semantic_model,
        gaussian_model,
        semantic_align,
    ).to(device)

    return full_model, renderer


# ------------------------------------------------
# Force model parameters to fp32
# ------------------------------------------------
def force_fp32(module):

    for m in module.modules():

        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data = m.weight.data.float()

        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data = m.bias.data.float()

    return module


# ------------------------------------------------
# Main training
# ------------------------------------------------
def train():
    cfg = load_config(
        "/home/wdong/workspace/4DSSGT/configs/config.yaml"
    )


    os.makedirs("outputs/stage1", exist_ok=True)
    os.makedirs("outputs/stage2", exist_ok=True)

    print("stage1_iters =", cfg["train"]["stage1_iters"])
    print("stage2_iters =", cfg["train"]["stage2_iters"])
    stage1_psnr = []
    stage1_acc = []

    stage2_psnr = []
    stage2_acc = []


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, renderer = build_all(cfg, device)

    model = force_fp32(model)
    model = model.float()

    trainer = End2EndTrainer(
        model=model,
        renderer=renderer,
        device=device,
        stage1_iters=cfg["train"]["stage1_iters"],
        stage2_iters=cfg["train"]["stage2_iters"],
    )

    stage1_loader, stage2_loader = build_dataloaders(
        cfg["data"]["video_frames"],
        cfg["data"]["semantic_paths"],
        batch_size=cfg["train"]["batch_size"],
        start_frame=cfg["data"]["start_frame"],
        end_frame=cfg["data"]["end_frame"],

    )

    print("\n========== DATA INFO ==========")
    print("Stage1 dataset size:", len(stage1_loader.dataset))
    print("Stage1 loader length:", len(stage1_loader))
    print("Stage2 dataset size:", len(stage2_loader.dataset))
    print("Stage2 loader length:", len(stage2_loader))
    print("===============================\n")

    # ==========================================================
    # Stage 1
    # ==========================================================

    print("========== STAGE 1 ==========")

    stage1_iter = cycle(stage1_loader)

    stage1_bar = tqdm(
        range(cfg["train"]["stage1_iters"]),
        desc="Stage1",
        dynamic_ncols=True,
    )

    stage1_start = time.time()

    for step in stage1_bar:

        batch = next(stage1_iter)

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        loss, psnr, acc, render_out = trainer.train_step_stage1(batch)

        stage1_bar.set_postfix(
            loss=f"{loss:.4f}",
            psnr=f"{psnr:.2f}",
            acc=f"{acc:.4f}",
        )
        stage1_psnr.append(psnr)
        stage1_acc.append(acc)

        if step % 1000 == 0:
            gt = batch["target_rgb"][0, 0].detach().cpu()

            pred = render_out["rgb"][0, 0].detach().cpu()

            pred = (pred + 1.0) / 2.0

            compare = torch.cat(
                [gt, pred],
                dim=2,
            )

            torchvision.utils.save_image(
                gt,
                f"outputs/stage1/{step:06d}_gt.png",
            )

            torchvision.utils.save_image(
                pred,
                f"outputs/stage1/{step:06d}_render.png",
            )

            torchvision.utils.save_image(
                compare,
                f"outputs/stage1/{step:06d}_compare.png",
            )

            sem_pred = (
                render_out["semantic"][0, 0]
                .argmax(dim=0)
                .cpu()
                .long()
                )

            color_pred = COLOR_MAP[sem_pred]

            sem_pred = (
                    color_pred
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
            )

            sem_gt = batch["semantic_mask"][0].cpu().long()

            color_gt = COLOR_MAP[sem_gt]

            sem_gt = (
                    color_gt
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
            )


            torchvision.utils.save_image(
                sem_gt.unsqueeze(0),
                f"outputs/stage1/{step:06d}_sem_gt.png",
            )


            torchvision.utils.save_image(
                sem_pred.unsqueeze(0),
                f"outputs/stage1/{step:06d}_sem_pred.png",
            )

            torchvision.utils.save_image(
                sem_gt.unsqueeze(0),
                f"outputs/stage1/{step:06d}_sem_gt.png",
            )
    stage1_time = time.time() - stage1_start

    print(
        f"\nStage1 Finished "
        f"| Time: {stage1_time / 60:.2f} min "
        f"| Avg PSNR: {sum(stage1_psnr) / len(stage1_psnr):.2f}"
    )
    # ==========================================================
    # Switch to Stage 2
    # ==========================================================

    init_batch = next(iter(stage2_loader))

    init_batch = {
        k: v.to(device)
        for k, v in init_batch.items()
    }

    trainer.switch_to_stage2(init_batch)

    print("\n========== STAGE 2 ==========")

    # ==========================================================
    # Stage 2
    # ==========================================================

    stage2_iter = cycle(stage2_loader)

    stage2_bar = tqdm(
        range(cfg["train"]["stage2_iters"]),
        desc="Stage2",
        dynamic_ncols=True,
    )

    stage2_start = time.time()

    for step in stage2_bar:

        batch = next(stage2_iter)

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        loss, psnr, acc, render_out = trainer.train_step_stage2(
            batch,
            step,
        )

        stage2_bar.set_postfix(
            loss=f"{loss:.4f}",
            psnr=f"{psnr:.2f}",
            acc=f"{acc:.4f}",
        )
        stage2_psnr.append(psnr)
        stage2_acc.append(acc)
        if step % 100 == 0:

            print(
                f"[Stage2] "
                f"Step {step}/{cfg['train']['stage2_iters']} "
                f"| Loss {loss:.4f} "
                f"| PSNR {psnr:.2f} "
                f"| Acc {acc}"
            )
        if step % 1000 == 0:
            gt = batch["target_rgb"][0, 0].detach().cpu()

            pred = render_out["rgb"][0, 0].detach().cpu()

            pred = (pred + 1.0) / 2.0

            compare = torch.cat(
                [gt, pred],
                dim=2,
            )

            torchvision.utils.save_image(
                gt,
                f"outputs/stage2/{step:06d}_gt.png",
            )

            torchvision.utils.save_image(
                pred,
                f"outputs/stage2/{step:06d}_render.png",
            )

            torchvision.utils.save_image(
                compare,
                f"outputs/stage2/{step:06d}_compare.png",
            )

            sem_pred = (
                render_out["semantic"][0, 0]
                .argmax(dim=0)
                .cpu()
                .long()
            )

            color_pred = COLOR_MAP[sem_pred]

            sem_pred = (
                    color_pred
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
            )

            sem_gt = batch["semantic_mask"][0].cpu().long()

            color_gt = COLOR_MAP[sem_gt]

            sem_gt = (
                    color_gt
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
            )

            torchvision.utils.save_image(
                sem_pred.unsqueeze(0),
                f"outputs/stage2/{step:06d}_sem_pred.png",
            )

            torchvision.utils.save_image(
                sem_gt.unsqueeze(0),
                f"outputs/stage2/{step:06d}_sem_gt.png",
            )
    stage2_time = time.time() - stage2_start

    print(
        f"\nStage2 Finished "
        f"| Time: {stage2_time / 60:.2f} min "
        f"| Avg PSNR: {sum(stage2_psnr) / len(stage2_psnr):.2f}"
    )
    import matplotlib.pyplot as plt

    # PSNR
    plt.figure(figsize=(10, 5))

    plt.plot(stage1_psnr, label="Stage1")
    plt.plot(stage2_psnr, label="Stage2")

    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("PSNR During Training")
    plt.legend()
    plt.grid(True)

    plt.savefig("psnr_curve.png", dpi=300)
    plt.close()

    # ACC
    plt.figure(figsize=(10, 5))

    plt.plot(stage1_acc, label="Stage1")
    plt.plot(stage2_acc, label="Stage2")

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy During Training")
    plt.legend()
    plt.grid(True)

    plt.savefig("acc_curve.png", dpi=300)
    plt.close()

    print("\nSaving final model...")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        "outputs/final_model.pth"
    )

    print("Model saved to outputs/final_model.pth")


# ------------------------------------------------
# Entry
# ------------------------------------------------
if __name__ == "__main__":
    train()