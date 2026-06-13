"""End-to-end torch pipeline as a class: SAM 3 video + DeepLabCut pose.

    SAM3Video.stream  ->  SegmentResult (boxes/masks/track-ids, GPU tensors)
                          |-> DLCTorchPose.predict_tensor (on-GPU crop -> keypoints)
                          |-> sv.Detections (masks/boxes) for annotation
    annotate (masks + boxes + track id + keypoints) -> mp4

Uses the HF (Transformers) SAM 3 video backend. The raw, pre-NMS boxes and the
on-device frame feed pose directly, so cropping stays on the GPU SAM 3 ran on;
NMS is not applied — the annotated video shows every raw detection.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from deeplabsam.pose.dlc import DLCTorchPose
from deeplabsam.segment.sam3video import SAM3Video


def rgb_frames(
    cap: cv2.VideoCapture, max_frames: int | None = None
) -> Iterator[np.ndarray]:
    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        n += 1


class Pipeline:
    """SAM 3 video (HF, detect + track) + DeepLabCut pose, end to end.

    Build once (loads both models), then call :meth:`run` per video. ``device``
    and ``super_animal`` default to auto-CUDA and the top-view mouse head.
    """

    def __init__(
        self, device: str | None = None, super_animal: str = "superanimal_topviewmouse"
    ):
        self.predictor = SAM3Video(device=device)
        self.pose_head = DLCTorchPose(super_animal=super_animal, device=device)

    def run(
        self,
        video_path: str,
        text: str | list[str] = "mouse",
        max_frames: int | None = None,
        output_dir: str = "outputs",
        name_suffix: str = "_annotated",
        keypoint_threshold: float = 0.3,
    ) -> Path:
        """Stream ``video_path``, detect+track+pose, write an annotated mp4.

        The output is named after the input video (``<stem><name_suffix>.mp4``)
        and written into ``output_dir``, which is created if needed. Returns the
        output path.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")
        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if max_frames:
            total = min(total, max_frames) if total else max_frames

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(video_path).stem}{name_suffix}.mp4"

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (W, H)
        )
        mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        vertex_annot = sv.VertexAnnotator(color=sv.Color.RED, radius=3)

        stream = self.predictor.stream(rgb_frames(cap, max_frames), text)
        for res in tqdm(stream, total=total):
            annotated = cv2.cvtColor(res.frame, cv2.COLOR_RGB2BGR)
            if not res:
                writer.write(annotated)
                continue

            # Raw, pre-NMS boxes + the on-GPU frame feed pose directly: the frame
            # is uploaded once and boxes already live on the inference device, so
            # the letterbox + pose run on the GPU SAM 3 ran on. Per-instance masks
            # are passed too, so each crop keeps only the target animal's pixels.
            kpts = self.pose_head.predict_tensor(
                res.frame_tensor(), res.boxes, res.masks
            )
            xy = kpts[:, :, :2].copy()
            conf = kpts[:, :, 2]
            xy[conf < keypoint_threshold] = 0  # let annotator skip them
            keypoints = sv.KeyPoints(xy=xy, confidence=conf)

            det = res.to_detections()
            annotated = mask_annot.annotate(annotated, det)
            annotated = box_annot.annotate(annotated, det)
            labels = [f"#{tid}" for tid in det.tracker_id]
            annotated = label_annot.annotate(annotated, det, labels=labels)
            annotated = vertex_annot.annotate(annotated, keypoints)
            writer.write(annotated)

        cap.release()
        writer.release()
        print(f"Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    pipe = Pipeline()
    pipe.run(
        video_path="/home/juan/Videos/edit.mp4",
        text="mice",
        max_frames=100,
        output_dir="outputs",
    )
