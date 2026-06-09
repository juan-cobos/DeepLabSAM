"""Shared per-frame result for the SAM 3 video segment stage.

The HF backend (``sam3video``) yields this :class:`SegmentResult`, keeping
detections as GPU tensors so the pose stage (a torch DeepLabCut head in the same
CUDA process) can crop on-device with no host round-trip: :meth:`frame_tensor`
uploads the frame once and ``boxes`` is already on the inference device.

The raw, pre-NMS boxes feed pose directly. :meth:`to_detections` builds numpy
``sv.Detections`` for annotation / serialization (also pre-NMS — no dedup).
"""

from dataclasses import dataclass, field

import numpy as np
import supervision as sv
import torch


@dataclass
class SegmentResult:
    """One frame of SAM 3 video output, kept in two representations.

    The pose stage consumes the GPU tensors directly: :meth:`frame_tensor`
    uploads the frame once and ``boxes`` is already on-device, so cropping
    happens on the GPU with no per-crop host round-trip (see
    ``DLCTorchPose.predict_tensor``). Boxes are the raw, pre-NMS detections — the
    caller applies NMS only when it wants it.

    Attributes:
        frame_idx: index of this frame in the stream.
        frame: original RGB ``(H, W, 3)`` uint8 array (source for pose crops).
        boxes: ``(N, 4)`` xyxy float tensor on the inference device.
        masks: ``(N, H, W)`` bool tensor on the inference device.
        object_ids: ``(N,)`` int — persistent track IDs.
        scores: ``(N,)`` float tensor on the inference device.
        class_names: length-``N`` list of the prompt text that detected each object.
        class_ids: ``(N,)`` int — index of each object's prompt in the prompt list.
    """

    frame_idx: int
    frame: np.ndarray
    boxes: torch.Tensor
    masks: torch.Tensor
    object_ids: np.ndarray
    scores: torch.Tensor
    class_names: list[str]
    class_ids: np.ndarray
    # Lazily-uploaded GPU copy of ``frame`` (CHW uint8); cached across calls.
    _frame_tensor: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __len__(self):
        return len(self.object_ids)

    def frame_tensor(self, device=None) -> torch.Tensor:
        """``frame`` as a ``(3, H, W)`` uint8 tensor on ``device`` (cached).

        Defaults to the device of ``boxes`` (the inference device), so the pose
        stage can crop on the same GPU SAM 3 ran on. The upload happens once per
        frame; repeated calls return the cached tensor.
        """
        if device is None:
            device = self.boxes.device
        if self._frame_tensor is None or self._frame_tensor.device != torch.device(
            device
        ):
            self._frame_tensor = (
                torch.from_numpy(np.ascontiguousarray(self.frame))
                .to(device)
                .permute(2, 0, 1)
            )
        return self._frame_tensor

    def to_detections(self) -> sv.Detections:
        """Numpy ``sv.Detections`` for annotation / serialization.

        Carries ``tracker_id`` (object IDs), ``class_id`` + ``data["class_name"]``
        (the prompt), masks and confidences. This is the only place tensors are
        moved to the CPU. Pre-NMS — the raw detections, no dedup.
        """
        if len(self) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=self.boxes.detach().cpu().float().numpy().astype(np.float32),
            mask=self.masks.detach().cpu().numpy().astype(bool),
            confidence=self.scores.detach().cpu().float().numpy().astype(np.float32),
            tracker_id=self.object_ids,
            class_id=self.class_ids,
            data={"class_name": np.asarray(self.class_names)},
        )
