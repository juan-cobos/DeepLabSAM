"""End-to-end torch pipeline as a class: SAM 3 video + DeepLabCut pose.

    SAM3Video.stream  ->  SegmentResult (boxes/masks/track-ids, GPU tensors)
                          |-> DLCTorchPose.predict_tensor (on-GPU crop -> keypoints)
                          |-> sv.Detections (masks/boxes) for annotation
    annotate (masks + boxes + track id + keypoints) -> mp4

Uses the HF (Transformers) SAM 3 video backend. The raw, pre-NMS boxes and the
on-device frame feed pose directly, so cropping stays on the GPU SAM 3 ran on;
NMS is not applied — the annotated video shows every raw detection.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import supervision as sv
import torch
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
        # TF32 speeds the fp32 pose matmuls at no memory cost. (cuDNN benchmark
        # was tried and dropped: it ~tripled peak VRAM for no gain, since pose is
        # only ~7% of runtime — SAM 3's forward dominates.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.predictor = SAM3Video(device=device)
        self.pose_head = DLCTorchPose(super_animal=super_animal, device=device)

    def run(
        self,
        video_path: str,
        text: str | list[str] = "mouse",
        max_frames: int | None = None,
        output_dir: str | Path = "outputs",
        name_suffix: str = "_annotated",
        keypoint_threshold: float = 0.3,
        export_json: bool = True,
        profile: bool = False,
    ) -> Path:
        """Stream ``video_path``, detect+track+pose, write an annotated mp4.

        The output is named after the input video (``<stem><name_suffix>.mp4``)
        and written into ``output_dir``, which is created if needed. When
        ``export_json`` is set, per-frame detections (boxes, scores, track + class
        ids, each tagged with its frame index) are also written to a sibling
        ``<stem><name_suffix>.json`` via ``sv.JSONSink``. Returns the mp4 path.

        With ``profile``, each frame's time is split into detect (SAM 3 forward),
        pose, and io (annotate + encode + json) and a breakdown is printed; this
        adds CUDA syncs, so leave it off for the fastest run.
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
        json_path = output_path.with_suffix(".json")

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (W, H)
        )
        mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        vertex_annot = sv.VertexAnnotator(color=sv.Color.RED, radius=3)

        # JSONSink writes the accumulated detections to disk on context exit; a
        # nullcontext keeps one code path when JSON export is off.
        sink = sv.JSONSink(str(json_path)) if export_json else contextlib.nullcontext()

        on_cuda = self.predictor.device == "cuda"

        def sync():
            if profile and on_cuda:
                torch.cuda.synchronize()

        prof = {"detect": 0.0, "pose": 0.0, "io": 0.0}
        stream = self.predictor.stream(rgb_frames(cap, max_frames), text)
        with sink:
            mark = perf_counter()
            for res in tqdm(stream, total=total):
                sync()
                prof["detect"] += perf_counter() - mark  # time to yield this frame

                annotated = cv2.cvtColor(res.frame, cv2.COLOR_RGB2BGR)
                if not res:
                    writer.write(annotated)
                    mark = perf_counter()
                    continue

                t_pose = perf_counter()
                kpts = self.pose_head.predict_tensor(
                    res.frame_tensor(), res.boxes, res.masks
                )
                xy = kpts[:, :, :2].copy()
                conf = kpts[:, :, 2]
                xy[conf < keypoint_threshold] = 0  # let annotator skip them
                keypoints = sv.KeyPoints(xy=xy, confidence=conf)
                sync()
                prof["pose"] += perf_counter() - t_pose

                t_io = perf_counter()
                det = res.to_detections()
                if export_json:
                    sink.append(det, custom_data={"frame_index": res.frame_idx})
                annotated = mask_annot.annotate(annotated, det)
                annotated = box_annot.annotate(annotated, det)
                labels = [
                    f"{cls} #{tid}"
                    for cls, tid in zip(det.data["class_name"], det.tracker_id)
                ]
                annotated = label_annot.annotate(annotated, det, labels=labels)
                annotated = vertex_annot.annotate(annotated, keypoints)
                writer.write(annotated)
                prof["io"] += perf_counter() - t_io
                mark = perf_counter()

        cap.release()
        writer.release()
        print(f"Saved: {output_path}" + (f" + {json_path}" if export_json else ""))
        if profile:
            tot = sum(prof.values()) or 1.0
            print("profile (s, % of total):")
            for stage, secs in prof.items():
                print(f"  {stage:7s} {secs:7.2f}  {100 * secs / tot:5.1f}%")
        return output_path


if __name__ == "__main__":
    pipe = Pipeline()
    pipe.run(
        video_path="/home/juan/Videos/edit.mp4",
        text="mice",
        max_frames=100,
        output_dir="outputs",
    )
