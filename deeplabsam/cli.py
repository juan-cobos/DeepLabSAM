"""Typer CLI for the DeepLabSAM pipeline.

Exposes every knob of the end-to-end run — the SAM 3 video model build
(``image_size``, detection threshold, memory window), the DLC pose head
(``super_animal``, pose model, device, crop size) and the per-run options
(prompts, frame cap, output, thresholds, JSON export, mask dump) — as flags, so a full run
is one command:

    deeplabsam video.mp4 --text mouse --image-size 1008 --max-frames 100
"""

from pathlib import Path
from typing import Annotated, Optional

import typer


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
        Optional[int],
        typer.Option(help="Cap the number of frames processed (default: whole video)."),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for the annotated mp4 (and JSON).")
    ] = Path("outputs"),
    name_suffix: Annotated[
        str, typer.Option(help="Suffix appended to the input stem for outputs.")
    ] = "_annotated",
    keypoint_threshold: Annotated[
        float,
        typer.Option(
            min=0.0, max=1.0, help="Min keypoint confidence to draw a vertex."
        ),
    ] = 0.3,
    nms_threshold: Annotated[
        float,
        typer.Option(
            min=0.0, max=1.0, help="Per-frame NMS box-IoU threshold for duplicates."
        ),
    ] = 0.5,
    class_agnostic: Annotated[
        bool,
        typer.Option(help="NMS across classes (drop overlaps regardless of prompt)."),
    ] = False,
    export_json: Annotated[
        bool, typer.Option(help="Also write per-frame detections + keypoints JSON.")
    ] = True,
    save_masks: Annotated[
        bool,
        typer.Option(
            help="Also dump each frame's masks as .npy into a per-video "
            "<output-dir>/<video-stem>_masks folder."
        ),
    ] = False,
    # --- SAM 3 video model build ---------------------------------------------
    image_size: Annotated[
        int,
        typer.Option(
            help="SAM 3 input resolution (speed/accuracy knob; lower = faster)."
        ),
    ] = 1008,
    score_threshold: Annotated[
        Optional[float],
        typer.Option(
            min=0.0,
            max=1.0,
            help="SAM 3 detection score threshold (overrides config default).",
        ),
    ] = None,
    memory_window: Annotated[
        int, typer.Option(help="SAM 3 tracking memory window (frames).")
    ] = 64,
    # --- DLC pose head build --------------------------------------------------
    super_animal: Annotated[
        str, typer.Option(help="SuperAnimal pose model id.")
    ] = "superanimal_topviewmouse",
    pose_model: Annotated[
        Optional[str],
        typer.Option(help="DLC pose backbone (default chosen per super-animal)."),
    ] = None,
    pose_input_size: Annotated[
        int,
        typer.Option(help="DLC pose crop letterbox size (multiple of pad divisor)."),
    ] = 256,
    device: Annotated[
        Optional[str], typer.Option(help="Torch device for pose (default: auto).")
    ] = None,
) -> None:
    """Run a video through SAM 3 detect+track and DeepLabCut pose."""
    # Imports are deferred so `--help` stays instant (the model imports are heavy).
    from transformers import Sam3VideoConfig

    from deeplabsam.pipeline import Pipeline
    from deeplabsam.pose.backends.dlc import DLCPoseHead
    from deeplabsam.segment.sam3video import Sam3VideoWrapper

    config = Sam3VideoConfig.from_pretrained("facebook/sam3")
    config.image_size = image_size
    if score_threshold is not None:
        config.score_threshold_detection = score_threshold

    predictor = Sam3VideoWrapper(config=config, memory_window=memory_window)
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


def main() -> None:
    """Console-script entry point: run the pipeline as a single command."""
    typer.run(run)


if __name__ == "__main__":
    main()
