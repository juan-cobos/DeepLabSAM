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
        video_path: str | Path,
        text: str | list[str] = "mouse",
        max_frames: int | None = None,
        output_dir: str | Path = "outputs",
        name_suffix: str = "_annotated",
        keypoint_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        class_agnostic: bool = False,
        export_json: bool = True,
        save_masks: bool = False,
    ) -> Path:
        """Run ``video_path`` through detect+track+pose."""

        predictor = self.predictor.reset(text)

        decoder = VideoDecoder(video_path, dimension_order="NHWC", device="cuda")
        meta = decoder.metadata
        frame_rate = meta.average_fps or 30.0
        W, H = meta.width, meta.height
        total = meta.num_frames or len(decoder)
        if max_frames:
            total = min(total, max_frames) if total else max_frames

        video_path = Path(video_path)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}{name_suffix}.mp4"
        json_path = output_path.with_suffix(".json")
        masks_dir = output_dir / f"{video_path.stem}_masks"

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (W, H)
        )
        mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        vertex_annot = sv.VertexAnnotator(color=sv.Color.RED, radius=3)
        sink = sv.JSONSink(str(json_path)) if export_json else contextlib.nullcontext()

        # SAM 3's pre-NMS object ids leave holes (e.g. 0, 1, 3) in the labels and
        # TRACK-keyed colors when NMS suppresses a tracklet every frame. Remap
        # survivors to contiguous display ids, first-seen order; reset per video.
        display_ids: dict[int, int] = {}

        def to_display_ids(raw_ids: np.ndarray) -> np.ndarray:
            return np.array(
                [display_ids.setdefault(int(r), len(display_ids)) for r in raw_ids],
                dtype=int,
            )

        with sink:
            for frame_idx, frame in enumerate(tqdm(decoder, total=total)):
                if max_frames and frame_idx >= max_frames:
                    break
                # HWC uint8 RGB tensor; fed as-is to the model and the pose head.
                # The annotators draw on a BGR numpy view — the one conversion
                # cv2's writer actually needs.
                result = predictor(frame)
                if save_masks:
                    result.save_masks(masks_dir, frame_idx=frame_idx)
                det = result.to_detections()  # carries class_id + data["class_name"]
                annotated = cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_RGB2BGR)
                if len(det) == 0:
                    writer.write(annotated)
                    continue

                det = det.with_nms(threshold=nms_threshold, class_agnostic=class_agnostic)
                # Contiguous display ids so labels/colors skip NMS's id holes.
                det.tracker_id = to_display_ids(det.tracker_id)
                class_names = list(det.data["class_name"])

                # Per-detection prompt splits animals (pose targets) from objects
                # (mask/track only). Boolean over detection order.
                is_animal = np.array(
                    [any(a in p.lower() for a in SUPPORTED_ANIMALS) for p in class_names]
                )

                # Pose runs only on animal detections; object masks pass through
                # untouched. Keypoints are scattered back into full detection order
                # so JSON/annotation stay aligned (objects keep all-zero, hence
                # not-visible, keypoints).
                kpts = np.zeros((len(det), self.pose_head.num_bodyparts, 3), dtype=np.float32)
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
        return output_path
