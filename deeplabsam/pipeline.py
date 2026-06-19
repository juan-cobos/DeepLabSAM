import contextlib
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from transformers import Sam3VideoConfig

from deeplabsam.pose import PoseHead
from deeplabsam.pose.backends.dlc import DLCPoseHead
from deeplabsam.segment.sam3video import Sam3VideoWrapper

# A prompt that *contains* one of these (case-insensitive substring, so
# "black mouse" matches) gets DLC pose; any other prompt (e.g. "object") is
# tracked/masked only, so pose is never fitted onto an object.
SUPPORTED_ANIMALS = frozenset({"mouse", "mice", "animal"})


class Pipeline:
    """SAM 3 video (HF, detect + track) + DeepLabCut pose, end to end.

    Takes an already-built segmentation ``predictor`` and ``pose_head``, so all
    model configuration (SAM 3 ``config``, pose ``super_animal``/``device``) lives
    where those components are constructed — the pipeline just wires them together.
    Each :meth:`run` calls ``predictor.reset(text)`` to start a fresh tracking
    session, so one pipeline processes many videos without reloading the models.
    """

    def __init__(self, predictor: Sam3VideoWrapper, pose_head: PoseHead):
        # TF32 speeds the fp32 pose matmuls at no memory cost. (cuDNN benchmark
        # was tried and dropped: it ~tripled peak VRAM for no gain, since pose is
        # only ~7% of runtime — SAM 3's forward dominates.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.predictor = predictor.eval()
        self.pose_head = pose_head

    @classmethod
    def default(
        cls,
        *,
        image_size: int = 1008,
        super_animal: str = "superanimal_topviewmouse",
        device: str | None = None,
    ) -> "Pipeline":
        """Build a standard pipeline without hand-wiring the components.

        Convenience for the common case: a SAM 3 wrapper at the given
        ``image_size`` (the speed/accuracy knob) and a DLC pose head for
        ``super_animal``. Drop below 1008 to trade accuracy for speed. For
        anything else (custom config, swapped pose backend, test fakes),
        construct ``predictor``/``pose_head`` yourself and call ``Pipeline(...)``.
        """
        config = Sam3VideoConfig.from_pretrained("facebook/sam3")
        config.image_size = image_size
        return cls(
            Sam3VideoWrapper(config=config),
            DLCPoseHead(super_animal=super_animal, device=device),
        )

    def run(
        self,
        video_path: str,
        text: str | list[str] = "mouse",
        max_frames: int | None = None,
        output_dir: str | Path = "outputs",
        name_suffix: str = "_annotated",
        keypoint_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        export_json: bool = True,
    ) -> Path:
        """Run ``video_path`` through detect+track+pose, write an annotated mp4.

        ``text`` is the full prompt set — pass animals and objects together, e.g.
        ``["mouse", "object"]``. Detections whose prompt is in ``SUPPORTED_ANIMALS``
        are run through DLC pose; any other prompt (the object) is tracked/masked
        only, so pose is never fitted onto an object and downstream code can still
        quantify animal-object interaction from the object masks.

        ``nms_threshold`` applies class-agnostic non-max suppression per frame:
        with overlapping prompts (e.g. ``["mouse", "rat"]``) the same animal can be
        matched by both, so duplicates with box IoU above this threshold are
        dropped, keeping the higher-confidence detection.

        The output is named after the input video (``<stem><name_suffix>.mp4``)
        and written into ``output_dir``, which is created if needed. When
        ``export_json`` is set, per-frame detections (boxes, scores, track + class
        ids, each tagged with its frame index and prompt ``class_name``) are also
        written to a sibling ``<stem><name_suffix>.json`` via ``sv.JSONSink``.
        Returns the mp4 path.
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
                det = result.to_detections()  # carries class_id + data["class_name"]
                annotated = cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_RGB2BGR)
                if len(det) == 0:
                    writer.write(annotated)
                    continue
                # Class-agnostic so cross-prompt duplicates (one animal matched by
                # both "mouse" and "rat") collapse to one. NMS reorders/filters, so
                # ``det`` is the single source of truth from here on — read the
                # prompt back from the class_name it carries, not from ``result``.
                det = det.with_nms(threshold=nms_threshold, class_agnostic=True)
                class_names = list(det.data["class_name"])

                # Per-detection prompt splits animals (pose targets) from objects
                # (mask/track only). Boolean over detection order.
                is_animal = np.array(
                    [
                        any(a in p.lower() for a in SUPPORTED_ANIMALS)
                        for p in class_names
                    ]
                )

                # Pose runs only on animal detections; object masks pass through
                # untouched. Keypoints are scattered back into full detection order
                # so JSON/annotation stay aligned (objects keep all-zero, hence
                # not-visible, keypoints).
                kpts = np.zeros(
                    (len(det), self.pose_head.num_bodyparts, 3), dtype=np.float32
                )
                if is_animal.any():
                    kpts[is_animal] = self.pose_head.predict_tensor(
                        frame, det.xyxy[is_animal], det.mask[is_animal]
                    )
                conf = kpts[:, :, 2]
                # `visible` skips sub-threshold vertices in the annotator.
                keypoints = sv.KeyPoints(
                    xy=kpts[:, :, :2],
                    keypoint_confidence=conf,
                    visible=conf >= keypoint_threshold,
                )
                if export_json:
                    # det carries boxes/scores/ids/class_name; kpts (N, K, 3) as a
                    # length-N list rides in custom_data, sliced per detection.
                    sink.append(
                        det,
                        custom_data={
                            "frame_index": frame_idx,
                            "keypoints": kpts.tolist(),
                        },
                    )
                annotated = mask_annot.annotate(annotated, det)
                annotated = box_annot.annotate(annotated, det)
                # With one prompt every label is just "<prompt> #<id>"; with
                # [animal, object] prompts each detection shows its own class.
                labels = [f"{p} #{tid}" for p, tid in zip(class_names, det.tracker_id)]
                annotated = label_annot.annotate(annotated, det, labels=labels)
                annotated = vertex_annot.annotate(annotated, keypoints)
                writer.write(annotated)

        writer.release()
        print(f"Saved: {output_path}" + (f" + {json_path}" if export_json else ""))
        return output_path


if __name__ == "__main__":
    from time import perf_counter

    max_frames = 10
    pipe = Pipeline.default()

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
