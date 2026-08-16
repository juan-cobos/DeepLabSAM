"""The pose-head contract the pipeline depends on.

A pose backend turns SAM 3 detections (per-instance boxes + masks) into
keypoints. The pipeline only needs the two members below, so it's typed against
this ``Protocol`` rather than any concrete backend — any object that satisfies it
(DLC today, SLEAP/Lightning Pose/ONNX later) is interchangeable. ``Protocol`` is
used deliberately at this seam so third-party wrappers don't have to subclass.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class PoseHead(Protocol):
    #: Number of keypoints the head predicts (the ``K`` in the output shape).
    num_bodyparts: int

    def predict_tensor(
        self,
        images: torch.Tensor,
        boxes: np.ndarray | torch.Tensor,
        masks: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Estimate keypoints for each detection.

        ``images`` is the frame(s) to pose on (named plural since backends may
        accept a batch). Returns an ``(N, K, 3)`` array of ``(x, y, confidence)``
        in original-image pixel coords, aligned to the input ``boxes`` order.
        """
        ...
