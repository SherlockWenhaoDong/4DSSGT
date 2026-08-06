import os
import torch
from tqdm import tqdm

from src.models.encoders.Semantic_encoder import SAM3LoRAEncoder
from src.Dataloader.VideoDataset import VideoSemanticDataset


SAVE_DIR = "semantic_cache"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


def main():

    device = "cuda"

    encoder = SAM3LoRAEncoder(
        sam3_config="configs/sam3_lora.yaml",
        lora_weights="outputs/sam3_lora_full/best_lora_weights.pt",
        freeze=True,
    ).to(device)

    encoder.eval()

    dataset = YourDataset(
        split="train"
    )

    print("dataset size =", len(dataset))

    for idx in tqdm(range(len(dataset))):

        sample = dataset[idx]

        image = sample["image"]

        frame_id = sample["frame_id"]

        save_path = os.path.join(
            SAVE_DIR,
            f"{frame_id}.pt"
        )

        if os.path.exists(save_path):
            continue

        if image.ndim == 3:
            pass

        elif image.ndim == 4:
            image = image[0]

        image = image.to(device)

        with torch.no_grad():

            semantic_map = encoder.predict_all_masks(
                image
            )

            # [12,288,288]

            semantic_map = semantic_map.half()

        torch.save(
            semantic_map.cpu(),
            save_path
        )

    print("Done")


if __name__ == "__main__":
    main()