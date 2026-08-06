#!/usr/bin/env python3

import os
import cv2
import numpy as np
from tqdm import tqdm

from infer_sam import SAM3LoRAInference

# ==========================================================
# PATHS
# ==========================================================

IMAGE_DIR = "/home/wdong/workspace/sam3/frames/PBP01_01"

OUTPUT_DIR = (
    "/home/wdong/workspace/sam3/frames/PBP01_01_semantic"
)

CONFIG_PATH = (
    "outputs/sam3_lora_full/full_lora_config.yaml"
)

LORA_WEIGHTS = (
    "outputs/sam3_lora_full/best_lora_weights.pt"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================================
# CLASSES
# ==========================================================

CLASS_NAMES = [
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

# background = 0

CLASS_COLORS = {
    0: (255, 128, 0),

    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),

    4: (255, 255, 0),
    5: (255, 0, 255),

    6: (0, 255, 255),
    7: (128, 0, 0),

    8: (0, 128, 0),
    9: (0, 0, 128),

    10: (128, 128, 0),
    11: (128, 0, 128),
}


# ==========================================================
# COLORIZE
# ==========================================================

def colorize(label_map):

    h, w = label_map.shape

    color = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    for cls_id, rgb in CLASS_COLORS.items():

        color[label_map == cls_id] = rgb

    return color


# ==========================================================
# BUILD MODEL
# ==========================================================

print("Loading SAM3 + LoRA...")

inferencer = SAM3LoRAInference(
    config_path=CONFIG_PATH,
    weights_path=LORA_WEIGHTS,
    resolution=1008,
    detection_threshold=0.5,
    nms_iou_threshold=0.5,
)

print("SAM3 Loaded")


# ==========================================================
# PROCESS ONE IMAGE
# ==========================================================

def generate_semantic_map(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            f"Cannot open image: {image_path}"
        )

    h, w = image.shape[:2]

    semantic_id = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    try:

        results = inferencer.predict(
            image_path,
            CLASS_NAMES,
        )

        # --------------------------------------------------
        # prompt idx
        # --------------------------------------------------

        for idx in range(len(CLASS_NAMES)):

            result = results[idx]

            cls_id = idx + 1

            if result["num_detections"] == 0:
                continue

            masks = result["masks"]
            scores = result["scores"]

            if masks is None:
                continue

            if scores is None:
                continue

            if len(scores) == 0:
                continue

            best_idx = int(
                np.argmax(scores)
            )

            best_mask = masks[best_idx]

            semantic_id[best_mask] = cls_id

    except Exception as e:

        print(
            f"[WARNING] {os.path.basename(image_path)} "
            f"failed: {e}"
        )

        semantic_id[:] = 0

    return semantic_id


# ==========================================================
# MAIN
# ==========================================================

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        )
    )
])

print(f"Found {len(image_files)} images")

for fname in tqdm(image_files):

    image_path = os.path.join(
        IMAGE_DIR,
        fname,
    )

    try:

        semantic_id = generate_semantic_map(
            image_path
        )

        color_mask = colorize(
            semantic_id
        )

    except Exception as e:

        print(
            f"[ERROR] {fname}: {e}"
        )

        img = cv2.imread(image_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        color_mask = np.zeros(
            (h, w, 3),
            dtype=np.uint8
        )

    save_name = (
        os.path.splitext(fname)[0]
        + ".png"
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        save_name,
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            color_mask,
            cv2.COLOR_RGB2BGR
        )
    )

print("\nDone.")
print("Output:")
print(OUTPUT_DIR)