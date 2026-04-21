from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DTYPE = np.float32
_MEAN = np.array([0.485, 0.456, 0.406], dtype=DTYPE)
_STD = np.array([0.229, 0.224, 0.225], dtype=DTYPE)
_MEAN_255 = tuple((_MEAN * 255).tolist())
_MEAN_CHW = _MEAN.reshape(3, 1, 1)
_INV_STD_CHW = (1.0 / _STD).reshape(3, 1, 1)


class TVMInference:
    "Top View Mouse Inference"

    INPUT_SIZE = 256

    def __init__(self, cache_dir="dlc"):
        self.cache_dir = Path(cache_dir)
        model_path = self.cache_dir / "pose_hrnet_w32.onnx"
        if not model_path.exists():
            self.cache_dir.mkdir(exist_ok=True)
            self._download()

        providers = [
            p
            for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if p in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(str(model_path), providers=providers)

    def _download(self):
        import urllib.request
        import zipfile

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/JCobosAlvarez/DeepLabCut-TopViewMouse-onnx/resolve/main/hrnet_w32.zip"
        urllib.request.urlretrieve(url, "hrnet_w32.zip")
        with zipfile.ZipFile("hrnet_w32.zip") as z:
            z.extractall(self.cache_dir)

    def _preprocess(self, image_bgr, boxes):
        """Letterbox each box-crop into (N, 3, S, S); return batch + per-box transform."""
        size = self.INPUT_SIZE
        N = len(boxes)
        H_img, W_img = image_bgr.shape[:2]
        # Pad with normalized-black so crop padding matches the mean-subtracted region after *= _INV_STD_CHW.
        batch = np.empty((N, 3, size, size), dtype=DTYPE)
        batch[:] = -_MEAN_CHW
        transforms = np.zeros((N, 5), dtype=DTYPE)  # x1, y1, scale, pad_x, pad_y
        valid_idxs = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W_img, x2), min(H_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_bgr[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            scale = min(size / cw, size / ch)
            new_w, new_h = int(cw * scale), int(ch * scale)
            x_off, y_off = (size - new_w) // 2, (size - new_h) // 2
            blob = cv2.dnn.blobFromImage(
                crop, 1 / 255.0, (new_w, new_h), _MEAN_255, swapRB=True
            )
            batch[i, :, y_off : y_off + new_h, x_off : x_off + new_w] = blob[0]
            transforms[i] = (x1, y1, scale, x_off, y_off)
            valid_idxs.append(i)
        batch *= _INV_STD_CHW
        return batch, transforms, valid_idxs

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
        if len(boxes) == 0:
            return np.zeros((0, 0, 3), dtype=DTYPE)

        batch, transforms, valid_idxs = self._preprocess(image, boxes)
        if not valid_idxs:
            return np.zeros((len(boxes), 0, 3), dtype=DTYPE)

        kpts = self.session.run(["kpts"], {"image": batch[valid_idxs]})[0]
        tf = transforms[valid_idxs]
        xy = (kpts[:, :, :2] - tf[:, None, 3:]) / tf[:, None, 2:3] + tf[:, None, :2]

        out = np.zeros((len(boxes), kpts.shape[1], 3), dtype=DTYPE)
        out[valid_idxs] = np.concatenate([xy, kpts[:, :, 2:]], axis=2)
        return out
