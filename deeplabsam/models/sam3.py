from samexporter.sam3_onnx import SegmentAnything3ONNX
import numpy as np
from pathlib import Path
import cv2

class SAM3Inference:
    def __init__(self, cache_dir="sam3"):

        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self._download()

        encoder_model = self.cache_dir / "sam3_image_encoder.onnx"
        decoder_model = self.cache_dir / "sam3_decoder.onnx"
        language_encoder_model = self.cache_dir / "sam3_language_encoder.onnx"

        self.model = SegmentAnything3ONNX(
            encoder_model,
            decoder_model,
            language_encoder_model,
        )

    def _download(self):
        import urllib.request, zipfile
        url = "https://huggingface.co/vietanhdev/segment-anything-3-onnx-models/resolve/main/sam3_vit_h.zip"
        urllib.request.urlretrieve(url, "sam3_vit_h.zip")
        with zipfile.ZipFile("sam3_vit_h.zip") as z:
            z.extractall(self.cache_dir)

    def predict(self, image, text):
        """Run text-prompted segmentation.

        Returns:
            boxes: (N, 4) xyxy float32 — empty-mask rows are zero.
            masks: (N, H, W) bool — SAM3's per-instance channel dim is squeezed
                out here so callers (e.g. sv.Detections) can consume directly.
        """

        embedding = self.model.encode(image, text_prompt=text)
        # SAM3 returns (N, 1, H, W); drop the channel axis at the source.
        masks = self.model.predict_masks(embedding, text, confidence_threshold=0.5)[:, 0]

        # Derive xyxy boxes from mask extents (vectorized per-axis reductions).
        rows_any = masks.any(axis=2)  # (N, H)
        cols_any = masks.any(axis=1)  # (N, W)
        N = masks.shape[0]
        boxes = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            rs = np.where(rows_any[i])[0]
            cs = np.where(cols_any[i])[0]
            if rs.size and cs.size:
                boxes[i] = [cs[0], rs[0], cs[-1], rs[-1]]

        return boxes, masks
