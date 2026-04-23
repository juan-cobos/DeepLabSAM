import argparse


def _add_common_args(parser):
    parser.add_argument("--text", default="mice", help="OSAM text prompt")
    parser.add_argument("--sam-model", default="sam3:latest", help="osam model identifier")
    parser.add_argument("--pose-model", default="topviewmouse", help="DLC pose model")
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        default=None,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="optional box prompt as xyxy",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--max-annotations", type=int, default=100)
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.3,
        help="hide keypoints below this confidence",
    )


def main():
    parser = argparse.ArgumentParser(prog="deeplabsam")
    sub = parser.add_subparsers(dest="command", required=True)

    img = sub.add_parser("image", help="run inference on a single image")
    img.add_argument("image", help="input image path")
    img.add_argument("--output", default=None, help="output path (default: <input>_annotated.<ext>)")
    _add_common_args(img)

    vid = sub.add_parser("video", help="run inference on a video or webcam")
    vid.add_argument("video", nargs="?", default=None, help="video path (omit for webcam)")
    vid.add_argument("--output", default=None, help="output video path")
    vid.add_argument(
        "--json",
        default=None,
        help="annotations JSON path (default: <video>_annotations.json)",
    )
    _add_common_args(vid)

    sub.add_parser("list-models", help="list available pose and SAM models")

    args = parser.parse_args()

    if args.command == "list-models":
        from deeplabsam.models.dlc import _REGISTRY as _POSE
        from deeplabsam.models.sam import AVAILABLE_MODELS
        print("Pose models:")
        for name in _POSE:
            print(f"  {name}")
        print("\nSAM models:")
        for name in AVAILABLE_MODELS:
            print(f"  {name}")
    elif args.command == "image":
        from deeplabsam.runners.image import run
        run(args)
    else:
        from deeplabsam.runners.video import run
        run(args)
