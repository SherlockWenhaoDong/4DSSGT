import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import MODELS

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata,
)

from sam3.train.data.collator import collate_fn_api

from sam3.model.utils.misc import (
    copy_data_to_device,
)

from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

# LoRA
from lora_layers import (
    LoRAConfig,
    apply_lora_to_model,
    load_lora_weights,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@MODELS.register_module()
class SAM3LoRAEncoder(nn.Module):

    def __init__(
        self,
        sam3_config,
        lora_weights,
        freeze=True,
        resolution=1008,
        threshold=0.5,
    ):
        super().__init__()

        self.threshold = threshold
        self.resolution = resolution

        self.prompts = [
            "Prograsp forceps",
            "cloaca_inside",
            "cloaca_outside",
            "needle_driver_left",
            "needle_driver_right",
            "stomach_inside",
            "stomach_outside",
            "suture_needle_left",
            "suture_needle_right",
            "suture_wire_left",
            "suture_wire_right",
        ]

        self.sam3 = build_sam3_image_model(
            device="cuda",
            compile=False,
            load_from_HF=True,
            eval_mode=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        )

        self.load_lora(
            sam3_config,
            lora_weights,
        )

        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

        self.sam3.eval()

        if freeze:

            for p in self.sam3.parameters():
                p.requires_grad = False

    def load_lora(
            self,
            config_path,
            weights_path,
    ):

        import yaml

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        lora_cfg = cfg["lora"]

        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )

        self.sam3 = apply_lora_to_model(
            self.sam3,
            lora_config,
        )

        load_lora_weights(
            self.sam3,
            weights_path,
        )

    @torch.no_grad()
    def encode_image_once(self, pil_img):

        # 随便用一个prompt构建batch
        datapoint = self.create_datapoint_single(
            pil_img,
            self.prompts[0]
        )

        datapoint = self.transform(datapoint)

        batch = collate_fn_api(
            [datapoint],
            dict_key="input"
        )["input"]

        batch = copy_data_to_device(
            batch,
            next(self.parameters()).device
        )

        backbone_out = {
            "img_batch_all_stages": batch.img_batch
        }

        # 只执行一次最贵的ViT
        backbone_out.update(
            self.sam3.backbone.forward_image(
                batch.img_batch
            )
        )

        return backbone_out

    @torch.no_grad()
    def run_single_prompt_cached(
            self,
            pil_img,
            prompt,
            backbone_out,
    ):

        from sam3.model.geometry_encoders import Prompt

        datapoint = self.create_datapoint_single(
            pil_img,
            prompt
        )

        datapoint = self.transform(datapoint)

        batch = collate_fn_api(
            [datapoint],
            dict_key="input"
        )["input"]

        batch = copy_data_to_device(
            batch,
            next(self.parameters()).device
        )

        # 每个prompt只做text encode
        text_outputs = self.sam3.backbone.forward_text(
            batch.find_text_batch,
            device=next(self.parameters()).device
        )

        current_backbone_out = {
            **backbone_out,
            **text_outputs,
        }

        find_input = batch.find_inputs[0]
        find_target = batch.find_targets[0]

        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
        )

        out = self.sam3.forward_grounding(
            backbone_out=current_backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=geometric_prompt,
        )

        return out

    def create_datapoint(
            self,
            pil_image,
    ):

        w, h = pil_image.size

        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        queries = []

        for idx, prompt in enumerate(self.prompts):
            query = FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )

            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[sam_image]
        )

    def create_datapoint_single(
            self,
            pil_image,
            prompt,
    ):

        w, h = pil_image.size

        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        query = FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=0,
                original_image_id=0,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )

        return Datapoint(
            find_queries=[query],
            images=[sam_image]
        )

    @torch.no_grad()
    def predict_all_masks(self, img_tensor):

        from PIL import Image

        img = (
            img_tensor
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        img = (img * 255).astype("uint8")

        pil_img = Image.fromarray(img)

        # image encode

        with torch.autocast(
                "cuda",
                dtype=torch.bfloat16
        ):
            backbone_out = self.encode_image_once(
                pil_img
            )
        class_masks = []

        for prompt in self.prompts:
            with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16
            ):
                out = self.run_single_prompt_cached(
                    pil_img=pil_img,
                    prompt=prompt,
                    backbone_out=backbone_out,
                )


            pred_mask = out["pred_masks"]

            # [1,1,200,288,288]
            pred_mask = pred_mask.max(
                dim=2
            ).values

            pred_mask = (
                pred_mask
                .squeeze(0)
                .squeeze(0)
                .float()
            )

            class_masks.append(
                pred_mask
            )

        class_masks = torch.stack(
            class_masks,
            dim=0
        )

        foreground = class_masks.max(
            dim=0
        ).values

        background = (
                1.0 - foreground
        ).clamp(0.0, 1.0)

        semantic_map = torch.cat(
            [
                background.unsqueeze(0),
                class_masks
            ],
            dim=0
        )

        return semantic_map

    @torch.no_grad()
    def forward(self, x):

        # x: [B,3,T,H,W]

        B, C, T, H, W = x.shape

        batch_semantic = []

        for b in range(B):

            frame_semantic = []

            for t in range(T):
                img = x[b, :, t]

                sem = self.predict_all_masks(img)

                # [12,288,288]

                frame_semantic.append(sem)

            frame_semantic = torch.stack(
                frame_semantic,
                dim=0
            )

            # [T,12,288,288]

            batch_semantic.append(
                frame_semantic
            )

        batch_semantic = torch.stack(
            batch_semantic,
            dim=0
        )

        # [B,T,12,288,288]

        if T == 1:

            batch_semantic = batch_semantic.squeeze(1)

            # [B,12,288,288]

        else:

            batch_semantic = batch_semantic.mean(dim=1)

            # [B,12,288,288]

        return batch_semantic