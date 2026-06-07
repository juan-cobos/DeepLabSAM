"""Torch DeepLabCut SuperAnimal pose heads — no ``deeplabcut`` install required.

Uses the genuine DeepLabCut ``PoseModel`` code, vendored (and trimmed) under
``deeplabsam.pose``, plus the bundled configs +
``modelzoo_utils`` to fetch snapshots from the HF Model Zoo. Because it's DLC's
own model + predictor, the heatmap decode is parity-correct out of the box — no
need to reverse-engineer it.
"""

import numpy as np
import torch
from torchvision.transforms.v2 import functional as TF

from deeplabsam.pose.models import PoseModel
from deeplabsam.pose.modelzoo_utils import (
    get_super_animal_snapshot_path,
    load_super_animal_config,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

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
        self._mean = _IMAGENET_MEAN
        self._std = _IMAGENET_STD

    @torch.no_grad()
    def predict(self, image, boxes):
        """Pose on each box — numpy convenience wrapper around :meth:`predict_tensor`.

        Args:
            image: HxWx3 RGB uint8 array.
            boxes: (N, 4) xyxy in image pixel coords.

        Returns:
            (N, K, 3) array of (x, y, confidence) in original image coords.
        """
        img = torch.from_numpy(np.ascontiguousarray(image))
        return self.predict_tensor(img, torch.as_tensor(np.asarray(boxes)))

    @torch.no_grad()
    def predict_tensor(self, image, boxes):
        """Pose on each box, cropping on-device with torchvision transforms v2.

        The letterbox (aspect-preserving resize + center-pad) is built from
        ``v2.functional.resized_crop`` + ``pad``, so the frame and boxes SAM 3
        already produced on the GPU stay there until the (tiny) keypoint tensor
        comes back — no cv2, no per-crop host round-trip.

        Args:
            image: full frame as a ``(3, H, W)`` (or ``(H, W, 3)``) uint8/float
                tensor; moved to the model device and scaled to ``[0, 1]``.
            boxes: ``(N, 4)`` xyxy tensor in image pixel coords.

        Returns:
            (N, K, 3) array of (x, y, confidence) in original image coords.
        """
        if image.ndim == 3 and image.shape[0] != 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)  # HWC -> CHW
        image = TF.to_dtype(image.to(self.device), torch.float32, scale=True)
        boxes = torch.as_tensor(boxes, device=self.device).float().reshape(-1, 4)
        N = boxes.shape[0]
        if N == 0:
            return np.zeros((0, self.num_bodyparts, 3), dtype=np.float32)

        H, W = image.shape[-2:]
        size = self.INPUT_SIZE
        crops, tfs, valid = [], [], []
        for i, (x1, y1, x2, y2) in enumerate(boxes.round().to(torch.int64).tolist()):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cw, ch = x2 - x1, y2 - y1
            scale = min(size / cw, size / ch)
            new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
            pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
            # crop the box and aspect-preserving resize in one op, then center-pad
            # to a square SxS canvas ([left, top, right, bottom] padding).
            resized = TF.resized_crop(
                image, y1, x1, ch, cw, [new_h, new_w], antialias=True
            )
            canvas = TF.pad(
                resized, [pad_x, pad_y, size - new_w - pad_x, size - new_h - pad_y]
            )
            crops.append(canvas)
            tfs.append((x1, y1, scale, pad_x, pad_y))
            valid.append(i)

        if not valid:
            return np.zeros((N, self.num_bodyparts, 3), dtype=np.float32)

        batch = TF.normalize(torch.stack(crops), mean=self._mean, std=self._std)
        return self._decode(batch, tfs, valid, N)

    def _decode(self, batch, tfs, valid, N):
        """Run the model on a normalized crop batch and map poses to image coords.

        ``batch`` is ``(V, 3, S, S)`` on device; ``tfs`` holds the per-crop
        ``(x1, y1, scale, pad_x, pad_y)`` letterbox transform; ``valid`` are the
        box indices that produced a crop. Returns ``(N, K, 3)`` in image coords.
        """
        outputs = self.model(batch)
        preds = self.model.get_predictions(outputs)
        poses = preds["bodypart"]["poses"][:, 0].cpu().numpy()  # (V, K, 3), S coords

        tf = np.array(tfs, dtype=np.float32)  # (V, 5): x1, y1, scale, pad_x, pad_y
        xy = (poses[:, :, :2] - tf[:, None, 3:]) / tf[:, None, 2:3] + tf[:, None, :2]
        out = np.zeros((N, self.num_bodyparts, 3), dtype=np.float32)
        out[np.asarray(valid, dtype=int)] = np.concatenate(
            [xy, poses[:, :, 2:]], axis=2
        )
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    super_animal = "superanimal_topviewmouse"
    head = DLCTorchPose(super_animal=super_animal, device=device)
    print(
        f"Superanimal {super_animal}: {head.num_bodyparts} bodyparts ({head.model_name}) on {device}"
    )
