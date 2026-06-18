import contextlib
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from deeplabsam.pose.dlc import DLCTorchPose
from deeplabsam.segment.sam3video import Sam3VideoWrapper


def outputs_to_detections(result: dict) -> sv.Detections:
    """Sam3VideoWrapper output dict -> numpy ``sv.Detections`` for annotation/JSON.

    Carries boxes, masks, confidences and ``tracker_id`` (object IDs). This is the
    only place the GPU tensors are moved to the CPU.
    """
    object_ids = result["object_ids"].detach().cpu().numpy().astype(int)
    if len(object_ids) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=result["boxes"].detach().cpu().float().numpy().astype(np.float32),
        mask=result["masks"].detach().cpu().numpy().astype(bool),
        confidence=result["scores"].detach().cpu().float().numpy().astype(np.float32),
        tracker_id=object_ids,
    )


class Pipeline:
    """SAM 3 video (HF, detect + track) + DeepLabCut pose, end to end.

    Both models load once here; each :meth:`run` calls ``predictor.reset(text)`` to
    start a fresh tracking session, so one pipeline processes many videos without
    reloading SAM 3. ``device`` and ``super_animal`` default to auto-CUDA and the
    top-view mouse head.
    """

    def __init__(
        self,
        device: str | None = None,
        super_animal: str = "superanimal_topviewmouse",
        image_size: int = 1008,
    ):
        # TF32 speeds the fp32 pose matmuls at no memory cost. (cuDNN benchmark
        # was tried and dropped: it ~tripled peak VRAM for no gain, since pose is
        # only ~7% of runtime — SAM 3's forward dominates.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # SAM 3's forward scales ~quadratically with input resolution; drop
        # image_size below the 1008 default to trade accuracy for speed.
        self.predictor = Sam3VideoWrapper(image_size=image_size).eval()
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
    ) -> Path:
        """Run ``video_path`` through detect+track+pose, write an annotated mp4.

        The output is named after the input video (``<stem><name_suffix>.mp4``)
        and written into ``output_dir``, which is created if needed. When
        ``export_json`` is set, per-frame detections (boxes, scores, track + class
        ids, each tagged with its frame index) are also written to a sibling
        ``<stem><name_suffix>.json`` via ``sv.JSONSink``. Returns the mp4 path.
        """
        # Fresh tracking session for this video (model weights stay loaded).
        predictor = self.predictor.reset(text)

        # torchcodec decodes straight to torch tensors. NHWC (H, W, 3) is the
        # layout the SAM 3 processor and the pose head already want, so frames
        # reach both without a permute. Metadata gives fps / size / frame count
        # up front for the writer and the progress bar.
        decoder = VideoDecoder(video_path, dimension_order="NHWC", device="cuda")
        meta = decoder.metadata
        frame_rate = meta.average_fps or 30.0
        W, H = meta.width, meta.height
        total = meta.num_frames or len(decoder)
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
        sink = sv.JSONSink(str(json_path)) if export_json else contextlib.nullcontext()

        with sink:
            for frame_idx, frame in enumerate(tqdm(decoder, total=total)):
                if max_frames and frame_idx >= max_frames:
                    break
                # HWC uint8 RGB tensor; fed as-is to the model and the pose head.
                # The annotators draw on a BGR numpy view — the one conversion
                # cv2's writer actually needs.
                result = predictor(frame)
                det = outputs_to_detections(result)
                annotated = cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_RGB2BGR)
                if len(det) == 0:
                    writer.write(annotated)
                    continue

                kpts = self.pose_head.predict_tensor(
                    frame, result["boxes"], result["masks"]
                )
                conf = kpts[:, :, 2]
                # `visible` skips sub-threshold vertices in the annotator.
                keypoints = sv.KeyPoints(
                    xy=kpts[:, :, :2],
                    keypoint_confidence=conf,
                    visible=conf >= keypoint_threshold,
                )
                if export_json:
                    # kpts (N, K, 3) as a length-N list -> sink slices one
                    # (x, y, conf) keypoint set per detection.
                    sink.append(
                        det,
                        custom_data={
                            "frame_index": frame_idx,
                            "keypoints": kpts.tolist(),
                        },
                    )
                annotated = mask_annot.annotate(annotated, det)
                annotated = box_annot.annotate(annotated, det)
                # Per-object class = the prompt that found it (handles >1 prompt;
                # with a single prompt every label is just "<prompt> #<id>").
                obj_to_prompt = {
                    oid: p
                    for p, oids in result["prompt_to_obj_ids"].items()
                    for oid in oids
                }
                labels = [f"{obj_to_prompt[int(tid)]} #{tid}" for tid in det.tracker_id]
                annotated = label_annot.annotate(annotated, det, labels=labels)
                annotated = vertex_annot.annotate(annotated, keypoints)
                writer.write(annotated)

        writer.release()
        print(f"Saved: {output_path}" + (f" + {json_path}" if export_json else ""))
        return output_path


if __name__ == "__main__":
    from time import perf_counter

    max_frames = 1000
    pipe = Pipeline()

    t0 = perf_counter()
    pipe.run(
        video_path="/home/juan/Videos/edit.mp4",
        text="mice",
        max_frames=max_frames,
        output_dir="outputs",
        export_json=False,
    )
    dt = perf_counter() - t0
    print(f"{max_frames} frames in {dt:.2f}s = {max_frames / dt:.2f} fps")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() // 1024**2} MiB")
