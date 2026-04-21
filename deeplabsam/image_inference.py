import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from models.dlc import TVMInference
from models.sam3 import SAM3Inference

KEYPOINT_THRESHOLD = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="input image path")
    parser.add_argument("--text", default="mice", help="SAM3 text prompt")
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

    sam_model = SAM3Inference()
    pose_model = TVMInference()

    box_annot = sv.BoxAnnotator()
    mask_annot = sv.MaskAnnotator()
    vertex_annot = sv.VertexAnnotator(color=sv.Color.RED, radius=3)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, masks = sam_model.predict(frame_rgb, text=args.text)

    # TVMInference uses cv2.dnn.blobFromImage(swapRB=True) → expects BGR.
    kpts = pose_model.predict(frame, boxes)

    annotated = frame.copy()
    if len(boxes):
        detections = sv.Detections(
            xyxy=boxes,
            confidence=np.ones(len(boxes), dtype=np.float32),
            mask=masks,
        )
        annotated = mask_annot.annotate(annotated, detections)
        annotated = box_annot.annotate(annotated, detections)

    if kpts.size:
        xy = kpts[:, :, :2].copy()
        conf = kpts[:, :, 2]
        xy[conf < KEYPOINT_THRESHOLD] = 0
        annotated = vertex_annot.annotate(
            annotated, sv.KeyPoints(xy=xy, confidence=conf)
        )

    cv2.imwrite(str(out_path), annotated)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
