from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DTYPE = np.float32
_MEAN = np.array([0.485, 0.456, 0.406], dtype=DTYPE)
_STD = np.array([0.229, 0.224, 0.225], dtype=DTYPE)
# Normalized value of a black pixel — used for letterbox padding so the model
# sees the same padding distribution it was trained with.
_PAD_CHW = ((-_MEAN) / _STD).reshape(3, 1, 1)


@dataclass(frozen=True)
class _ModelConfig:
    url: str
    zip_name: str
    model_file: str
    input_size: int


_REGISTRY: dict[str, _ModelConfig] = {
    "topviewmouse": _ModelConfig(
        url="https://huggingface.co/JCobosAlvarez/DeepLabCut-TopViewMouse-onnx/resolve/main/hrnet_w32.zip",
        zip_name="hrnet_w32.zip",
        model_file="pose_hrnet_w32.onnx",
        input_size=256,
    ),
}


class DLCPose:
    """DeepLabCut pose estimator backed by an ONNX model from the registry."""

    def __init__(self, model="topviewmouse", cache_dir="dlc"):
        if model not in _REGISTRY:
            raise ValueError(f"Unknown model {model!r}. Available: {list(_REGISTRY)}")
        cfg = _REGISTRY[model]
        self.cfg = cfg
        self.INPUT_SIZE = cfg.input_size
        self.cache_dir = Path(cache_dir)
        model_path = self.cache_dir / cfg.model_file
        if not model_path.exists():
            self._download(cfg)

        providers = [
            p
            for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if p in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(str(model_path), providers=providers)

    def _download(self, cfg):
        import urllib.request
        import zipfile

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.cache_dir / cfg.zip_name
        urllib.request.urlretrieve(cfg.url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(self.cache_dir)
        zip_path.unlink()

    def _letterbox(self, crop):
        """Resize + center-pad a crop to (3, S, S) normalized CHW float32."""
        size = self.INPUT_SIZE
        ch, cw = crop.shape[:2]
        scale = min(size / cw, size / ch)
        new_w, new_h = int(cw * scale), int(ch * scale)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2

        rgb = cv2.cvtColor(cv2.resize(crop, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        chw = ((rgb.astype(DTYPE) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)

        out = np.broadcast_to(_PAD_CHW, (3, size, size)).copy()
        out[:, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = chw
        return out, scale, pad_x, pad_y

    def predict(self, image, boxes):
        """Pose estimation on each box.

        Args:
            image: HxWx3 BGR uint8.
            boxes: (N, 4) xyxy in image pixel coords.

        Returns:
            (N, K, 3) array of (x, y, confidence) in original image coords.
            Invalid/out-of-frame boxes yield zero rows.
        """
        boxes = np.asarray(boxes, dtype=DTYPE).reshape(-1, 4)
        N = len(boxes)
        if N == 0:
            return np.zeros((0, 0, 3), dtype=DTYPE)

        H, W = image.shape[:2]
        crops, tfs, valid_idxs = [], [], []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            chw, scale, pad_x, pad_y = self._letterbox(image[y1:y2, x1:x2])
            crops.append(chw)
            tfs.append((x1, y1, scale, pad_x, pad_y))
            valid_idxs.append(i)

        if not valid_idxs:
            return np.zeros((N, 0, 3), dtype=DTYPE)

        batch = np.stack(crops)
        poses = self.session.run(["poses"], {"image": batch})[0]

        tf = np.array(tfs, dtype=DTYPE)  # (V, 5): x1, y1, scale, pad_x, pad_y
        xy = (poses[:, :, :2] - tf[:, None, 3:]) / tf[:, None, 2:3] + tf[:, None, :2]

        out = np.zeros((N, poses.shape[1], 3), dtype=DTYPE)
        out[valid_idxs] = np.concatenate([xy, poses[:, :, 2:]], axis=2)
        return out


if __name__ == "__main__":
    image_path = "examples/images/example.png"
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read: {image_path}")

    model = DLCPose()
    boxes = np.array([[607, 450, 770, 726]], dtype=np.float32)
    kpts = model.predict(image, boxes)
    print(f"Keypoints: {kpts.shape}, max conf: {kpts[0, :, 2].max():.3f}")
