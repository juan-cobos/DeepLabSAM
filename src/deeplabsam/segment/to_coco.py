"""Video -> COCO: segment + track with SAM 3, write frames + a COCO annotation file.

The "segment" stage of the segment / curate / pose workflow: stream a video
through :class:`~deeplabsam.segment.sam3video.Sam3VideoWrapper` so each tracked
instance gets a standard COCO bbox/segmentation record, alongside the decoded
frames and (optionally) an annotated viz for a quick sanity check. Hand the
output off for manual curation before :func:`deeplabsam.pose.from_coco.pose_from_coco`
fits keypoints on it.
"""

import json
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image
from supervision.dataset.formats.coco import (
    classes_to_coco_categories,
    detections_to_coco_annotations,
)
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from transformers import Sam3VideoConfig

from deeplabsam.segment.sam3video import Sam3VideoWrapper


def _json_default(o):
    """Coerce numpy scalars/arrays (from supervision's bbox/area) to native types."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def segment_to_coco(
    video: str | Path,
    out_dir: str | Path,
    prompt: list[str] = ["mouse"],
    threshold: float = 0.0,
    max_detections: int | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    num_maskmem: int | None = 64,
    image_size: int | None = None,
    checkpoint_path: str | Path = "facebook/sam3",
    dtype: torch.dtype = torch.bfloat16,
    save_viz: bool = True,
) -> Path:
    """Track ``prompt`` through ``video`` with SAM 3 and write a COCO dataset.

    Writes ``out_dir/frames/frame_NNNNN.jpg`` (raw decoded frames), optionally
    ``out_dir/viz/frame_NNNNN.png`` (masks/boxes/labels drawn), and
    ``out_dir/annotations.json`` (COCO detections + segmentations, the SAM 3
    ``score``, and the persistent tracker ``track_id`` per instance).

    Returns:
        Path to the written ``annotations.json``.

    """
    out_dir = Path(out_dir)
    torch.set_float32_matmul_precision("high")

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if save_viz:
        viz_dir = out_dir / "viz"
        viz_dir.mkdir(exist_ok=True)
        mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

    print(f"Loading SAM 3 from {checkpoint_path} ...")
    config = Sam3VideoConfig.from_pretrained(str(checkpoint_path))
    if image_size is not None:
        config.image_size = image_size
    wrapper = Sam3VideoWrapper(
        text=prompt,
        checkpoint_path=checkpoint_path,
        config=config,
        dtype=dtype,
        num_maskmem=num_maskmem,
    )

    # Global, stable category ids from the prompts (SegmentResult.to_detections
    # re-derives class_id per frame from whichever prompts fired, so we remap by
    # class name to keep category ids consistent across the whole video).
    categories = classes_to_coco_categories(sorted(prompt))
    name_to_category_id = {c["name"]: c["id"] for c in categories}

    decoder = VideoDecoder(video, dimension_order="NHWC", device=wrapper.device)
    meta = decoder.metadata
    total = meta.num_frames or 0
    if max_frames is not None:
        total = min(total, max_frames) if total else max_frames

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    image_id = 1
    annotation_id = 1

    pbar = tqdm(decoder, total=total, desc="tracking")
    for frame_idx, frame in enumerate(pbar):
        if max_frames is not None and frame_idx >= max_frames:
            break

        # Every frame is streamed through the tracker to preserve memory/continuity,
        # but only every `stride` frame is written out.
        result = wrapper(frame)
        if frame_idx % stride != 0:
            continue

        rgb = frame.detach().cpu().numpy()  # (H, W, 3) uint8 RGB
        height, width = rgb.shape[:2]
        name = f"frame_{frame_idx:05d}"
        Image.fromarray(rgb).save(frames_dir / f"{name}.jpg")
        coco_images.append(
            {
                "id": image_id,
                "file_name": f"frames/{name}.jpg",
                "width": width,
                "height": height,
                "frame_index": frame_idx,
            },
        )

        detections = result.to_detections()
        if threshold > 0 and len(detections):
            detections = detections[detections.confidence >= threshold]
        # Keep only the K highest-scoring instances (e.g. the single animal).
        if max_detections is not None and len(detections) > max_detections:
            keep = np.argsort(detections.confidence)[::-1][:max_detections]
            detections = detections[keep]

        if len(detections):
            # Remap the per-frame class_id to the global, name-based category id.
            names = detections.data["class_name"]
            detections.class_id = np.array(
                [name_to_category_id[n] - 1 for n in names],
                dtype=int,
            )
            # COCO `area` is the *mask* area. supervision only reads it from
            # data["area"] (a round-trip hook for already-annotated files) and
            # otherwise falls back to the bounding-box area, which overstates a
            # thin/diagonal animal several-fold. Count the mask pixels instead.
            detections.data["area"] = detections.mask.sum(axis=(1, 2)).astype(float)
            anns, annotation_id = detections_to_coco_annotations(
                detections=detections,
                image_id=image_id,
                annotation_id=annotation_id,
                approximation_percentage=0.0,  # exact masks (RLE/polygon)
            )
            # anns are in detection order; attach SAM 3 score + persistent track id.
            for ann, score, tid in zip(
                anns,
                detections.confidence,
                detections.tracker_id,
            ):
                ann["score"] = round(float(score), 5)
                ann["track_id"] = int(tid)
            coco_annotations.extend(anns)

        if save_viz:
            scene = rgb[..., ::-1].copy()  # RGB -> BGR for the annotators
            if len(detections):
                labels = [
                    f"{n} #{t}"
                    for n, t in zip(
                        detections.data["class_name"],
                        detections.tracker_id,
                    )
                ]
                scene = mask_annot.annotate(scene, detections)
                scene = box_annot.annotate(scene, detections)
                scene = label_annot.annotate(scene, detections, labels=labels)
            Image.fromarray(scene[..., ::-1]).save(viz_dir / f"{name}.png")

        image_id += 1
        pbar.set_postfix(instances=len(coco_annotations))

    coco = {
        "info": {
            "description": f"SAM 3 video '{' + '.join(prompt)}' auto-annotations",
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    out_path = out_dir / "annotations.json"
    with open(out_path, "w") as f:
        json.dump(coco, f, default=_json_default)
    print(
        f"\nDone. {len(coco_images)} frames, {len(coco_annotations)} instances -> {out_path}",
    )
    return out_path
