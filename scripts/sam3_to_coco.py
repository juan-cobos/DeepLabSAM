"""Segment frames with the SAM 3 model and export COCO annotations."""

from __future__ import annotations

import enum
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import numpy as np
import supervision as sv
import torch
import typer
from PIL import Image
from supervision.dataset.formats.coco import (
    classes_to_coco_categories,
    detections_to_coco_annotations,
)
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class DType(enum.StrEnum):
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    """Seed every RNG and force deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _json_default(o):
    """Coerce numpy scalars/arrays (from supervision's bbox/area) to native types."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def iter_frames(
    video: Path | None,
    image_dir: Path | None,
    max_frames: int | None,
    stride: int,
) -> tuple[Iterator[tuple[int, str, np.ndarray]], int]:
    """Yield ``(index, name, rgb_uint8_hwc)`` for every kept frame."""
    if video is not None:
        decoder = VideoDecoder(video, dimension_order="NHWC")
        n = decoder.metadata.num_frames or 0
        if max_frames is not None:
            n = min(n, max_frames) if n else max_frames
        indices = range(0, n, stride)

        def gen():
            for i in indices:
                yield i, f"frame_{i:05d}", decoder[i].numpy()

        return gen(), len(indices)

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    paths = paths[:max_frames][::stride]

    def gen():
        for i, path in enumerate(paths):
            yield i, path.stem, np.asarray(Image.open(path).convert("RGB"))

    return gen(), len(paths)


def segment(
    model: Sam3Model,
    processor: Sam3Processor,
    prompts: list[str],
    category_ids: dict[str, int],
    rgb: np.ndarray,
    threshold: float,
    mask_threshold: float,
) -> sv.Detections:
    """Run every prompt on one image in a single batched forward pass."""
    inputs = processor(
        images=[rgb] * len(prompts),
        text=prompts,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs["original_sizes"].tolist(),
    )

    per_prompt = []
    for prompt, res in zip(prompts, results):
        n = len(res["scores"])
        if n == 0:
            continue
        per_prompt.append(
            sv.Detections(
                xyxy=res["boxes"].float().cpu().numpy(),
                mask=res["masks"].bool().cpu().numpy(),
                confidence=res["scores"].float().cpu().numpy(),
                # supervision class ids are 0-based; COCO category ids are 1-based.
                class_id=np.full(n, category_ids[prompt] - 1, dtype=int),
                data={"class_name": np.full(n, prompt, dtype=object)},
            ),
        )
    return sv.Detections.merge(per_prompt) if per_prompt else sv.Detections.empty()


def main(
    out_dir: Annotated[
        Path,
        typer.Option(help="Where to write annotations.json, frames/ and viz/."),
    ],
    video: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Input video file."),
    ] = None,
    image_dir: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Directory of input images."),
    ] = None,
    prompt: Annotated[
        list[str],
        typer.Option(
            help="SAM 3 text prompt / COCO category name; repeat for several, "
            "e.g. --prompt mouse --prompt 'human hand'.",
        ),
    ] = ["mouse"],
    threshold: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Drop instances scoring below this."),
    ] = 0.5,
    mask_threshold: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Probability threshold used to binarize the predicted masks.",
        ),
    ] = 0.5,
    nms_threshold: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Per-frame mask-IoU NMS threshold for duplicates (applied before --top-k).",
        ),
    ] = 0.5,
    max_detections: Annotated[
        int | None,
        typer.Option(
            "--max-detections",
            "--top-k",
            min=1,
            help="Keep only the K highest-scoring instances per frame, across all prompts "
            "(e.g. 1 when exactly one animal is present; lets you lower --threshold safely).",
        ),
    ] = None,
    max_frames: Annotated[
        int | None,
        typer.Option(min=1, help="Stop after this many input frames/images (debug)."),
    ] = None,
    stride: Annotated[
        int,
        typer.Option(
            min=1,
            help="Keep every Nth frame/image (skipped frames are never run through the model).",
        ),
    ] = 1,
    checkpoint_path: Annotated[
        str,
        typer.Option(help="SAM 3 checkpoint (HF id or local path)."),
    ] = "facebook/sam3",
    dtype: Annotated[DType, typer.Option(help="Compute precision.")] = DType.bf16,
    viz: Annotated[
        bool,
        typer.Option(help="Write the annotated viz/ frames."),
    ] = True,
    seed: Annotated[
        int,
        typer.Option(
            help="Seed for python/numpy/torch RNGs (also forces deterministic CUDA kernels).",
        ),
    ] = 0,
) -> None:
    """Segment frames with the SAM 3 image model and export COCO annotations."""
    if (video is None) == (image_dir is None):
        raise typer.BadParameter("pass exactly one of --video or --image-dir")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if viz:
        viz_dir = out_dir / "viz"
        viz_dir.mkdir(exist_ok=True)
        mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 from {checkpoint_path} ...")
    model = Sam3Model.from_pretrained(checkpoint_path, dtype=DTYPES[dtype.value])
    model = model.to(device).eval()
    processor = Sam3Processor.from_pretrained(checkpoint_path)

    categories = classes_to_coco_categories(sorted(prompt))
    category_ids = {c["name"]: c["id"] for c in categories}

    frames, total = iter_frames(video, image_dir, max_frames, stride)
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    image_id = 1
    annotation_id = 1

    pbar = tqdm(frames, total=total, desc="segmenting")
    for frame_idx, name, rgb in pbar:
        height, width = rgb.shape[:2]
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

        detections = segment(
            model,
            processor,
            prompt,
            category_ids,
            rgb,
            threshold,
            mask_threshold,
        )
        detections = detections.with_nms(threshold=nms_threshold)
        # Keep only the K highest-scoring instances (e.g. the single animal).
        if max_detections is not None and len(detections) > max_detections:
            keep = np.argsort(detections.confidence)[::-1][:max_detections]
            detections = detections[keep]

        if len(detections):
            # COCO `area` is the *mask* area; supervision otherwise falls back to
            # the bounding-box area (see sam3video_to_coco.py).
            detections.data["area"] = detections.mask.sum(axis=(1, 2)).astype(float)
            anns, annotation_id = detections_to_coco_annotations(
                detections=detections,
                image_id=image_id,
                annotation_id=annotation_id,
                approximation_percentage=0.0,  # exact masks (RLE/polygon)
            )
            for ann, score in zip(anns, detections.confidence):
                ann["score"] = round(float(score), 5)
            coco_annotations.extend(anns)

        if viz:
            scene = rgb[..., ::-1].copy()  # RGB -> BGR for the annotators
            if len(detections):
                labels = [
                    f"{n} {s:.2f}"
                    for n, s in zip(detections.data["class_name"], detections.confidence)
                ]
                scene = mask_annot.annotate(scene, detections)
                scene = box_annot.annotate(scene, detections)
                scene = label_annot.annotate(scene, detections, labels=labels)
            Image.fromarray(scene[..., ::-1]).save(viz_dir / f"{name}.png")

        image_id += 1
        pbar.set_postfix(instances=len(coco_annotations))

    coco = {
        "info": {
            "description": f"SAM 3 image '{' + '.join(prompt)}' auto-annotations",
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


if __name__ == "__main__":
    typer.run(main)
