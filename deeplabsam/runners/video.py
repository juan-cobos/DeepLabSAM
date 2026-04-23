import argparse
from pathlib import Path

import cv2
import supervision as sv
from tqdm import tqdm
from trackers import OCSORTTracker

from deeplabsam import DeepLabSAM


def run(args):
    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video or 'webcam'}")

    if args.json is not None:
        json_path = Path(args.json)
    elif args.video is not None:
        in_path = Path(args.video)
        json_path = in_path.with_stem(in_path.stem + "_annotations").with_suffix(".json")
    else:
        json_path = Path("annotations.json")

    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = (
        cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (W, H)
        )
        if args.output
        else None
    )

    pipeline = DeepLabSAM(
        sam_model=args.sam_model,
        pose_model=args.pose_model,
        keypoint_threshold=args.keypoint_threshold,
        device=args.device,
    )
    tracker = OCSORTTracker(frame_rate=frame_rate)

    box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    vertex_annot = sv.VertexAnnotator(color=sv.Color.RED, radius=3)

    frame_idx = 0
    with sv.JSONSink(str(json_path)) as sink, tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = pipeline.detect(
                frame_rgb,
                text=args.text,
                boxes=args.box,
                iou_threshold=args.iou_threshold,
                score_threshold=args.score_threshold,
                max_annotations=args.max_annotations,
            )
            tracked = (
                tracker.update(detections) if len(detections) else sv.Detections.empty()
            )
            keypoints = pipeline.estimate_pose(frame, tracked)

            annotated = frame.copy()
            if len(tracked):
                annotated = mask_annot.annotate(annotated, tracked)
                annotated = box_annot.annotate(annotated, tracked)
                labels = [f"#{tid}" for tid in tracked.tracker_id]
                annotated = label_annot.annotate(annotated, tracked, labels=labels)
            if keypoints.xy.size:
                annotated = vertex_annot.annotate(annotated, keypoints)

            if writer is not None:
                writer.write(annotated)

            for i in range(len(tracked)):
                sink.append(
                    sv.Detections(
                        xyxy=tracked.xyxy[i : i + 1],
                        tracker_id=tracked.tracker_id[i : i + 1],
                    ),
                    custom_data={
                        "frame_idx": frame_idx,
                        "keypoints": keypoints.xy[i].tolist()
                        if i < len(keypoints.xy)
                        else [],
                    },
                )
            frame_idx += 1
            pbar.update()

    cap.release()
    if writer is not None:
        writer.release()
    print(f"Saved: {json_path}")
    if args.output:
        print(f"Saved: {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=None, help="video path (omit for webcam)")
    parser.add_argument("--text", default="mice")
    parser.add_argument("--sam-model", default="sam3:latest")
    parser.add_argument("--pose-model", default="topviewmouse")
    parser.add_argument(
        "--box", type=int, nargs=4, default=None, metavar=("XMIN", "YMIN", "XMAX", "YMAX")
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--max-annotations", type=int, default=100)
    parser.add_argument("--keypoint-threshold", type=float, default=0.3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--json", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
