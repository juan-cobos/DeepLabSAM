"""Torch DeepLabCut SuperAnimal pose heads — no ``deeplabcut`` install required.

Uses the genuine DeepLabCut ``PoseModel`` code, vendored (and trimmed) under
``deeplabsam.pose``, plus the bundled configs +
``modelzoo_utils`` to fetch snapshots from the HF Model Zoo. Because it's DLC's
own model + predictor, the heatmap decode is parity-correct out of the box — no
need to reverse-engineer it.
"""

import cv2
import numpy as np
import torch

from deeplabsam.pose.models import PoseModel
from deeplabsam.pose.modelzoo_utils import (
    get_super_animal_snapshot_path,
    load_super_animal_config,
)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Default torch pose model per SuperAnimal (HRNet where available; bird is
# ResNet-only). Both use the same HeatmapHead + HeatmapPredictor.
_DEFAULT_POSE_MODEL = {
    "superanimal_topviewmouse": "hrnet_w32",
    "superanimal_quadruped": "hrnet_w32",
    "superanimal_bird": "resnet_50",
}


class DLCTorchPose:
    INPUT_SIZE = 256  # divisible by 32 (HRNet auto-padding requirement)

    def __init__(
        self,
        super_animal="superanimal_topviewmouse",
        model_name=None,
        device=None,
    ):
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.super_animal = super_animal
        if model_name is None:
            model_name = _DEFAULT_POSE_MODEL.get(super_animal, "hrnet_w32")
        self.model_name = model_name

        config = load_super_animal_config(
            super_animal=super_animal,
            model_name=model_name,
            detector_name=None,
            device=device,
        )
        self.bodyparts = list(config["metadata"]["bodyparts"])
        self.num_bodyparts = len(self.bodyparts)

        self.model = PoseModel.build(config["model"])
        path = get_super_animal_snapshot_path(super_animal, model_name)
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{super_animal} load mismatch: missing={missing} unexpected={unexpected}"
            )

        self.model = self.model.to(device).eval()
        self._mean = _MEAN.to(device)
        self._std = _STD.to(device)

    def _letterbox(self, crop):
        """Resize + center-pad an RGB crop to (S, S, 3) uint8, with transform.

        DLC's ``colormode`` is RGB; the SAM 3 pipeline already yields RGB frames,
        so no color conversion is done here.
        """
        size = self.INPUT_SIZE
        ch, cw = crop.shape[:2]
        scale = min(size / cw, size / ch)
        new_w, new_h = int(cw * scale), int(ch * scale)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        resized = cv2.resize(crop, (new_w, new_h))
        out = np.zeros((size, size, 3), dtype=np.uint8)
        out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        return out, scale, pad_x, pad_y

    @torch.no_grad()
    def predict(self, image, boxes):
        """Pose on each box.

        Args:
            image: HxWx3 RGB uint8.
            boxes: (N, 4) xyxy in image pixel coords.

        Returns:
            (N, K, 3) array of (x, y, confidence) in original image coords.
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        N = len(boxes)
        if N == 0:
            return np.zeros((0, self.num_bodyparts, 3), dtype=np.float32)

        H, W = image.shape[:2]
        crops, tfs, valid = [], [], []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            lb, scale, pad_x, pad_y = self._letterbox(image[y1:y2, x1:x2])
            crops.append(lb)
            tfs.append((x1, y1, scale, pad_x, pad_y))
            valid.append(i)

        if not valid:
            return np.zeros((N, self.num_bodyparts, 3), dtype=np.float32)

        batch = torch.from_numpy(np.stack(crops)).to(self.device).permute(0, 3, 1, 2)
        batch = (batch.float() / 255.0 - self._mean) / self._std

        outputs = self.model(batch)
        preds = self.model.get_predictions(outputs)
        poses = preds["bodypart"]["poses"][:, 0].cpu().numpy()  # (V, K, 3), 256 coords

        tf = np.array(tfs, dtype=np.float32)  # (V, 5): x1, y1, scale, pad_x, pad_y
        xy = (poses[:, :, :2] - tf[:, None, 3:]) / tf[:, None, 2:3] + tf[:, None, :2]
        out = np.zeros((N, self.num_bodyparts, 3), dtype=np.float32)
        out[valid] = np.concatenate([xy, poses[:, :, 2:]], axis=2)
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    super_animal = "superanimal_topviewmouse"
    head = DLCTorchPose(super_animal=super_animal, device=device)
    print(
        f"Superanimal {super_animal}: {head.num_bodyparts} bodyparts ({head.model_name}) on {device}"
    )
