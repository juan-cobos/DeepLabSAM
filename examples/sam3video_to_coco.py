"""Segment a video with SAM 3 (detect + track) and export COCO annotations.

Video sibling of ``autolabel/sam3_to_coco.py``: instead of a folder of independent
images run through the SAM 3 *image* model, this streams a video through
``Sam3VideoWrapper`` so objects are tracked across frames. For every frame each
tracked instance is written to a standard COCO detection/segmentation file
(``annotations.json``) containing, per instance, a ``bbox``, a ``segmentation``
(exact RLE for masks with holes/multiple parts, polygon otherwise -- handled by
``supervision``), an ``area``, the SAM 3 ``score``, and the persistent
``track_id`` so instances can be followed across frames.

Alongside the annotations it writes:
  * ``frames/frame_00000.jpg`` -- the raw decoded frames (COCO ``file_name``),
  * ``viz/frame_00000.png``    -- the same frames with masks/boxes/labels drawn.

Example:
    python scripts/sam3video_to_coco.py \
        --video /home/juan/Videos/example_videos/edited/stereotypes.mp4 \
        --out-dir ./outputs/stereotypes \
        --prompt mouse --max-frames 300
"""

from __future__ import annotations

import argparse
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

from deeplabsam.segment.sam3video import Sam3VideoWrapper

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True, help="Input video file.")
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write annotations.json, frames/ and viz/.",
    )
    p.add_argument(
        "--prompt",
        nargs="+",
        default=["mouse"],
        help="One or more SAM 3 text prompts / COCO category names.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Drop instances scoring below this (0 keeps everything the tracker returns).",
    )
    p.add_argument(
        "--max-detections",
        "--top-k",
        dest="max_detections",
        type=int,
        default=None,
        help="Keep only the K highest-scoring instances per frame "
        "(e.g. 1 when exactly one animal is present; lets you lower --threshold safely).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many decoded frames (debug).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame (still streamed through the tracker in order).",
    )
    p.add_argument(
        "--num-maskmem",
        type=int,
        default=64,
        help="Memory window for eviction; -1 disables eviction (unbounded VRAM).",
    )
    p.add_argument(
        "--checkpoint-path",
        default="facebook/sam3",
        help="SAM 3 checkpoint (HF id or local path).",
    )
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Compute precision.",
    )
    p.add_argument(
        "--no-viz", action="store_true", help="Skip writing the annotated viz/ frames."
    )
    return p.parse_args()


def _json_default(o):
    """Coerce numpy scalars/arrays (from supervision's bbox/area) to native types."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    save_viz = not args.no_viz
    if save_viz:
        viz_dir = args.out_dir / "viz"
        viz_dir.mkdir(exist_ok=True)
        mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

    print(f"Loading SAM 3 from {args.checkpoint_path} ...")
    wrapper = Sam3VideoWrapper(
        text=args.prompt,
        checkpoint_path=args.checkpoint_path,
        dtype=DTYPES[args.dtype],
        num_maskmem=None if args.num_maskmem < 0 else args.num_maskmem,
    )

    # Global, stable category ids from the CLI prompts (SegmentResult.to_detections
    # re-derives class_id per frame from whichever prompts fired, so we remap by
    # class name to keep category ids consistent across the whole video).
    categories = classes_to_coco_categories(sorted(args.prompt))
    name_to_category_id = {c["name"]: c["id"] for c in categories}

    decoder = VideoDecoder(args.video, dimension_order="NHWC", device=wrapper.device)
    meta = decoder.metadata
    total = meta.num_frames or 0
    if args.max_frames is not None:
        total = min(total, args.max_frames) if total else args.max_frames

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    image_id = 1
    annotation_id = 1

    pbar = tqdm(decoder, total=total, desc="tracking")
    for frame_idx, frame in enumerate(pbar):
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        # Every frame is streamed through the tracker to preserve memory/continuity,
        # but only every --stride frame is written out.
        result = wrapper(frame)
        if frame_idx % args.stride != 0:
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
            }
        )

        detections = result.to_detections()
        if args.threshold > 0 and len(detections):
            detections = detections[detections.confidence >= args.threshold]
        # Keep only the K highest-scoring instances (e.g. the single animal).
        if args.max_detections is not None and len(detections) > args.max_detections:
            keep = np.argsort(detections.confidence)[::-1][: args.max_detections]
            detections = detections[keep]

        if len(detections):
            # Remap the per-frame class_id to the global, name-based category id.
            names = detections.data["class_name"]
            detections.class_id = np.array(
                [name_to_category_id[n] - 1 for n in names], dtype=int
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
                anns, detections.confidence, detections.tracker_id
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
                        detections.data["class_name"], detections.tracker_id
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
            "description": f"SAM 3 video '{' + '.join(args.prompt)}' auto-annotations"
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    out_path = args.out_dir / "annotations.json"
    with open(out_path, "w") as f:
        json.dump(coco, f, default=_json_default)
    print(
        f"\nDone. {len(coco_images)} frames, {len(coco_annotations)} instances -> {out_path}"
    )


if __name__ == "__main__":
    main()
