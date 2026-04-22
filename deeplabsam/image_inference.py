import argparse
from pathlib import Path

import cv2
import supervision as sv

from deeplabsam import DeepLabSAM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="input image path")
    parser.add_argument("--text", default="mice", help="OSAM text prompt")
    parser.add_argument(
        "--model", default="sam3:latest", help="osam model identifier"
    )
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        default=[607, 450, 770, 726],
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="box prompt as xyxy (defaults to example ROI)",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="osam Prompt iou_threshold"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="osam Prompt score_threshold",
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=100,
        help="osam Prompt max_annotations",
    )
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.3,
        help="hide keypoints below this confidence",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output image path (default: <input>_annotated.<ext>)",
    )
    args = parser.parse_args()

    in_path = Path(args.image)
    frame = cv2.imread(str(in_path))
    if frame is None:
        raise RuntimeError(f"Cannot read: {in_path}")
    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_stem(in_path.stem + "_annotated")
    )

    pipeline = DeepLabSAM(
        sam_model=args.model, keypoint_threshold=args.keypoint_threshold
    )

    box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    vertex_annot = sv.VertexAnnotator(color=sv.Color.RED, radius=3)

    detections, keypoints = pipeline.predict(
        frame,
        text=args.text,
        boxes=args.box,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_annotations=args.max_annotations,
    )
    print(f"Detections: {len(detections)}")

    annotated = frame.copy()
    if len(detections):
        annotated = mask_annot.annotate(annotated, detections)
        annotated = box_annot.annotate(annotated, detections)
    if keypoints.xy.size:
        annotated = vertex_annot.annotate(annotated, keypoints)

    cv2.imwrite(str(out_path), annotated)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
