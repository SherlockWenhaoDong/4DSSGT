import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os, glob
from PIL import Image


TOP_CROP = 85
BOTTOM_CROP = 80

BACKGROUND_COLOR = np.array(
    [255, 128, 0],
    dtype=np.uint8
)
CLASS_COLOR_MAP = {

    (255,128,0): 0,

    (255,0,0): 1,
    (0,255,0): 2,
    (0,0,255): 3,

    (255,255,0): 4,
    (255,0,255): 5,

    (0,255,255): 6,
    (128,0,0): 7,

    (0,128,0): 8,
    (0,0,128): 9,

    (128,128,0): 10,
    (128,0,128): 11,
}

# ---------------------------
# Camera utils
# ---------------------------
def build_default_intrinsics(H, W):
    f = max(H, W)
    return torch.tensor([
        [f, 0, W / 2],
        [0, f, H / 2],
        [0, 0, 1]
    ], dtype=torch.float32)


def build_fixed_extrinsics():
    return torch.eye(4, dtype=torch.float32)


def build_timestamp(idx, fps=30):
    return torch.tensor([idx / fps], dtype=torch.float32)




# ---------------------------
# Dataset
# ---------------------------
class VideoSemanticDataset(Dataset):
    def __init__(
        self,
        video_frames,
        semantic_paths,
        mode="stage1",
        fps=30,
        START_FRAME=10000,
        END_FRAME=30000
    ):
        if isinstance(video_frames, str):
            self.video = sorted(
                glob.glob(os.path.join(video_frames, "*.png")) +
                glob.glob(os.path.join(video_frames, "*.jpg"))
            )
        else:
            self.video = video_frames

        if isinstance(semantic_paths, str):
            self.semantic_paths = sorted(
                glob.glob(os.path.join(semantic_paths, "*.png")) +
                glob.glob(os.path.join(semantic_paths, "*.jpg"))
            )
        else:
            self.semantic_paths = semantic_paths

        self.semantic_paths = sorted(
            self.semantic_paths,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        self.video = sorted(
            self.video,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        self.video = self.video[START_FRAME:END_FRAME + 1]
        self.semantic_paths = self.semantic_paths[START_FRAME:END_FRAME + 1]

        assert len(self.video) == len(self.semantic_paths)
        assert len(self.video) > 0

        sample_img = np.array(
            Image.open(self.semantic_paths[0]).convert("RGB")
        )

        sample_img = sample_img[
                     TOP_CROP:sample_img.shape[0] - BOTTOM_CROP,
                     :
                     ]

        bg_mask = np.all(
            sample_img == [0, 0, 0],
            axis=-1
        )

        sample_img[bg_mask] = BACKGROUND_COLOR

        self.color_map = CLASS_COLOR_MAP


        self.fps = fps

        self.H, self.W = sample_img.shape[:2]

        self.K = build_default_intrinsics(self.H, self.W)
        self.RT = build_fixed_extrinsics()

        num_frames = len(self.video)

        if mode == "stage1":
            self.indices = np.arange(0, num_frames, 10)
        else:
            self.indices = np.arange(0, num_frames, 2)

    def load_mask(self, path):

        img = np.array(
            Image.open(path).convert("RGB")
        )

        img = img[
              TOP_CROP:img.shape[0] - BOTTOM_CROP,
              :
              ]

        bg_mask = np.all(
            img == [0, 0, 0],
            axis=-1
        )

        img[bg_mask] = BACKGROUND_COLOR

        img = cv2.resize(
            img,
            (512, 512),
            interpolation=cv2.INTER_NEAREST
        )

        H, W, _ = img.shape

        mask = np.zeros(
            (H, W),
            dtype=np.int64
        )

        for color, class_id in self.color_map.items():
            color = np.array(color)

            match = np.linalg.norm(
                img.astype(np.int32) - color.astype(np.int32),
                axis=-1
            ) < 5

            mask[match] = class_id

        return torch.from_numpy(mask).long()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        img = cv2.imread(self.video[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img[
              TOP_CROP:img.shape[0] - BOTTOM_CROP,
              :
              ]

        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)


        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        rgb = img

        # ✅ mask
        label = self.load_mask(self.semantic_paths[i])
        # label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_LINEAR)

        ts = build_timestamp(i, self.fps)

        return {
            "input_images": rgb.unsqueeze(0),  # [T=1,3,H,W]
            "input_timestamps": ts.unsqueeze(0),
            "input_intrinsics": self.K.unsqueeze(0),
            "input_extrinsics": self.RT.unsqueeze(0),

            "supervising_timestamps": ts.unsqueeze(0),
            "supervising_intrinsics": self.K.unsqueeze(0),
            "supervising_extrinsics": self.RT.unsqueeze(0),

            "target_rgb": rgb.unsqueeze(0),
            "semantic_mask": label,
        }


from torch.utils.data import DataLoader


def build_dataloaders(
    video_frames,
    semantic_paths,
    batch_size=1,
    start_frame=10000,
    end_frame=30000,
):

    stage1_loader = DataLoader(
        VideoSemanticDataset(video_frames, semantic_paths, "stage1", START_FRAME=start_frame,
        END_FRAME=end_frame),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    stage2_loader = DataLoader(
        VideoSemanticDataset(video_frames, semantic_paths, "stage2", START_FRAME=start_frame,
        END_FRAME=end_frame),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    return stage1_loader, stage2_loader