"""Typer CLI for DeepLabSAM.

Three subcommands cover the two ways to run the pipeline:

- ``deeplabsam run`` -- single-shot: video in, annotated mp4 (+ JSON) out.
- ``deeplabsam segment`` -- the first half of the curated workflow: video in,
  a COCO dataset out (frames, optional viz, ``annotations.json``) for manual
  curation before pose is fit.
- ``deeplabsam pose`` -- the second half: fits DeepLabCut pose onto a curated
  COCO export from ``segment``.

Every knob is a flag; run ``deeplabsam <command> --help`` for the full list.
``--help`` stays instant on every command -- the heavy model imports are
deferred until a run actually starts.
"""

import enum
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def run(
    video_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input video to detect + track + pose.",
        ),
    ],
    # --- prompts / run options ------------------------------------------------
    text: Annotated[
        list[str],
        typer.Option(
            "--text",
            "-t",
            help="Prompt(s); repeat for several, e.g. -t mouse -t object. "
            "Prompts containing mouse/mice/animal get pose; others are mask/track only.",
        ),
    ] = ["mouse"],
    max_frames: Annotated[
        int | None,
        typer.Option(help="Cap the number of frames processed (default: whole video)."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for the annotated mp4 (and JSON)."),
    ] = Path("outputs"),
    name_suffix: Annotated[
        str,
        typer.Option(help="Suffix appended to the input stem for outputs."),
    ] = "_annotated",
    keypoint_threshold: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Min keypoint confidence to draw a vertex.",
        ),
    ] = 0.3,
    nms_threshold: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Per-frame NMS box-IoU threshold for duplicates.",
        ),
    ] = 0.5,
    class_agnostic: Annotated[
        bool,
        typer.Option(help="NMS across classes (drop overlaps regardless of prompt)."),
    ] = False,
    export_json: Annotated[
        bool,
        typer.Option(help="Also write per-frame detections + keypoints JSON."),
    ] = True,
    save_masks: Annotated[
        bool,
        typer.Option(
            help="Also dump each frame's masks as .npy into a per-video "
            "<output-dir>/<video-stem>_masks folder.",
        ),
    ] = False,
    # --- SAM 3 video model build ---------------------------------------------
    checkpoint_path: Annotated[
        Path,
        typer.Option(
            help="Local directory (skips all Hugging Face requests) or HF repo id "
            "to load the SAM 3 video checkpoint from.",
        ),
    ] = Path("facebook/sam3"),
    image_size: Annotated[
        int,
        typer.Option(
            help="SAM 3 input resolution (speed/accuracy knob; lower = faster).",
        ),
    ] = 1008,
    score_threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help="SAM 3 detection score threshold (overrides config default).",
        ),
    ] = None,
    num_maskmem: Annotated[
        int,
        typer.Option(help="SAM 3 tracking memory window (frames)."),
    ] = 64,
    # --- DLC pose head build --------------------------------------------------
    super_animal: Annotated[
        str,
        typer.Option(help="SuperAnimal pose model id."),
    ] = "superanimal_topviewmouse",
    pose_model: Annotated[
        str | None,
        typer.Option(help="DLC pose backbone (default chosen per super-animal)."),
    ] = None,
    pose_input_size: Annotated[
        int,
        typer.Option(help="DLC pose crop letterbox size (multiple of pad divisor)."),
    ] = 256,
    device: Annotated[
        str | None,
        typer.Option(help="Torch device for pose (default: auto)."),
    ] = None,
) -> None:
    """Run a video through SAM 3 detect+track and DeepLabCut pose in one shot."""
    # Imports are deferred so `--help` stays instant (the model imports are heavy).
    from transformers import Sam3VideoConfig

    from deeplabsam.pipeline import Pipeline
    from deeplabsam.pose.backends.dlc import DLCPoseHead
    from deeplabsam.segment.sam3video import Sam3VideoWrapper

    config = Sam3VideoConfig.from_pretrained(checkpoint_path)
    config.image_size = image_size
    if score_threshold is not None:
        config.score_threshold_detection = score_threshold

    predictor = Sam3VideoWrapper(
        checkpoint_path=checkpoint_path,
        config=config,
        num_maskmem=num_maskmem,
    )
    pose_head = DLCPoseHead(
        super_animal=super_animal,
        model_name=pose_model,
        device=device,
        input_size=pose_input_size,
    )
    pipe = Pipeline(predictor=predictor, pose_head=pose_head)

    output_path = pipe.run(
        video_path=str(video_path),
        text=text,
        max_frames=max_frames,
        output_dir=output_dir,
        name_suffix=name_suffix,
        keypoint_threshold=keypoint_threshold,
        nms_threshold=nms_threshold,
        class_agnostic=class_agnostic,
        export_json=export_json,
        save_masks=save_masks,
    )
    typer.echo(f"Done. Annotated video written to {output_path}")


class Dtype(enum.StrEnum):
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"


@app.command()
def segment(
    video: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Input video file."),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Where to write annotations.json, frames/ and viz/."),
    ],
    prompt: Annotated[
        list[str],
        typer.Option(
            "--prompt",
            "-p",
            help="One or more SAM 3 text prompts / COCO category names.",
        ),
    ] = ["mouse"],
    threshold: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Drop instances scoring below this (0 keeps everything the tracker returns).",
        ),
    ] = 0.0,
    max_detections: Annotated[
        int | None,
        typer.Option(
            "--max-detections",
            "--top-k",
            help="Keep only the K highest-scoring instances per frame "
            "(e.g. 1 when exactly one animal is present; lets you lower --threshold safely).",
        ),
    ] = None,
    max_frames: Annotated[
        int | None,
        typer.Option(help="Stop after this many decoded frames (debug)."),
    ] = None,
    stride: Annotated[
        int,
        typer.Option(help="Keep every Nth frame (still streamed through the tracker in order)."),
    ] = 1,
    num_maskmem: Annotated[
        int,
        typer.Option(help="Memory window for eviction; -1 disables eviction (unbounded VRAM)."),
    ] = 64,
    image_size: Annotated[
        int | None,
        typer.Option(
            help="Override the checkpoint's inference resolution (square, px). "
            "Smaller is faster/less VRAM but less accurate; default keeps the model's native size.",
        ),
    ] = None,
    checkpoint_path: Annotated[
        Path,
        typer.Option(help="SAM 3 checkpoint (HF id or local path)."),
    ] = Path("facebook/sam3"),
    dtype: Annotated[
        Dtype,
        typer.Option(help="Compute precision."),
    ] = Dtype.bf16,
    viz: Annotated[
        bool,
        typer.Option(help="Write the annotated viz/ frames."),
    ] = True,
) -> None:
    """Segment + track a video with SAM 3 and export a COCO dataset for curation."""
    # Imports are deferred so `--help` stays instant (the model imports are heavy).
    import torch

    from deeplabsam.segment.to_coco import segment_to_coco

    dtypes = {Dtype.bf16: torch.bfloat16, Dtype.fp16: torch.float16, Dtype.fp32: torch.float32}
    out_path = segment_to_coco(
        video=video,
        out_dir=out_dir,
        prompt=prompt,
        threshold=threshold,
        max_detections=max_detections,
        max_frames=max_frames,
        stride=stride,
        num_maskmem=None if num_maskmem < 0 else num_maskmem,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        dtype=dtypes[dtype],
        save_viz=viz,
    )
    typer.echo(f"Done. Annotations written to {out_path}")


@app.command(name="pose")
def pose_command(
    in_path: Annotated[
        Path,
        typer.Option(
            "--in",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input COCO annotations.json with segmentations (from `deeplabsam segment`).",
        ),
    ],
    out_path: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Where to write the keypoint-augmented COCO file "
            "(default: <in>_pose.json next to the input).",
        ),
    ] = None,
    image_root: Annotated[
        Path | None,
        typer.Option(
            help="Directory the image `file_name` entries are relative to "
            "(default: the input file's directory).",
        ),
    ] = None,
    super_animal: Annotated[
        str,
        typer.Option(help="SuperAnimal pose model to fit."),
    ] = "superanimal_topviewmouse",
    model_name: Annotated[
        str | None,
        typer.Option(help="Backbone override (default: hrnet_w32, or resnet_50 for bird)."),
    ] = None,
    input_size: Annotated[
        int,
        typer.Option(
            help="Letterbox size fed to the pose model; must divide by the backbone's "
            "pad divisor (32 for HRNet).",
        ),
    ] = 256,
    device: Annotated[
        str | None,
        typer.Option(help="cuda / cpu (default: auto)."),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Keypoint confidence at or above which a keypoint counts as visible "
            "(COCO v=2) and is drawn in the video.",
        ),
    ] = 0.3,
    max_images: Annotated[
        int | None,
        typer.Option(help="Stop after this many images (debug)."),
    ] = None,
    video: Annotated[
        Path | None,
        typer.Option(
            help="Annotated mp4 to write for visual QC "
            "(default: <out>_check.mp4; use --no-video to skip).",
        ),
    ] = None,
    no_video: Annotated[
        bool,
        typer.Option("--no-video", help="Skip the QC video."),
    ] = False,
    fps: Annotated[
        float,
        typer.Option(help="QC video frame rate."),
    ] = 30.0,
    seed: Annotated[
        int,
        typer.Option(help="Seed for torch/numpy so repeated runs are byte-identical."),
    ] = 0,
) -> None:
    """Fit DeepLabCut pose onto a curated SAM 3 COCO export from `deeplabsam segment`."""
    # Imports are deferred so `--help` stays instant (the model imports are heavy).
    import numpy as np
    import torch

    from deeplabsam.pose.coco import pose_from_coco

    # Pose inference is already deterministic (eval mode, no sampling), but seeding
    # + pinning cuDNN's algorithm choice makes that a guarantee rather than a
    # property that a future backend change could quietly break.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out = pose_from_coco(
        in_path=in_path,
        out_path=out_path,
        image_root=image_root,
        super_animal=super_animal,
        model_name=model_name,
        input_size=input_size,
        device=device,
        threshold=threshold,
        max_images=max_images,
        video=video,
        save_video=not no_video,
        fps=fps,
    )
    typer.echo(f"Done. Keypoints written to {out}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
