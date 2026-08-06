import os
import cv2
import glob

# ------------------------------------------------
# ✅ 配置
# ------------------------------------------------
INPUT_IMG_DIR = "/home/wdong/workspace/4DGT/surgicaldata/images"
INPUT_MASK_DIR = "/home/wdong/workspace/nas_private/Data/stereo_reconstruction/Clip1_CAM2/viz"

OUTPUT_IMG_DIR = "/home/wdong/workspace/4DSSGT/Data/Clip1/images"
OUTPUT_MASK_DIR = "/home/wdong/workspace/4DSSGT/Data/Clip1/masks"

SIZE = (1024, 1024)

# ✅ 固定 ROI
ROI = (300, 80, 1620, 950)


# ------------------------------------------------
# ✅ 工具函数
# ------------------------------------------------
def crop_and_resize(img):
    x1, y1, x2, y2 = ROI

    img = img[y1:y2, x1:x2]
    img = cv2.resize(img, SIZE)

    return img


def process_folder(input_dir, output_dir):

    paths = sorted(
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg"))
    )

    print(f"Processing {len(paths)} files from {input_dir}")

    for i, p in enumerate(paths):

        img = cv2.imread(p)

        if img is None:
            print(f"⚠️ skip {p}")
            continue

        img = crop_and_resize(img)

        save_path = os.path.join(output_dir, os.path.basename(p))

        cv2.imwrite(save_path, img)

        print(f"[{i}/{len(paths)}] Done")

    print(f"✅ Saved to {output_dir}")


# ------------------------------------------------
# ✅ 主函数
# ------------------------------------------------
if __name__ == "__main__":

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    print("✅ Output directories ready:")
    print(OUTPUT_IMG_DIR)
    print(OUTPUT_MASK_DIR)

    print("\n=== Processing RGB ===")
    process_folder(INPUT_IMG_DIR, OUTPUT_IMG_DIR)

    print("\n=== Processing Semantic ===")
    process_folder(INPUT_MASK_DIR, OUTPUT_MASK_DIR)
