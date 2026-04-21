import argparse

import cv2
import numpy as np
import supervision as sv
from models.dlc import TVMInference
from models.sam3 import SAM3Inference
from trackers import OCSORTTracker

KEYPOINT_THRESHOLD = 0.3
DETECT_EVERY_N = 1
JSON_PATH = "annotations.json"
OUTPUT_VIDEO = "output.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="video path (omit for webcam)")
    parser.add_argument("--text", default="mice", help="SAM3 text prompt")
    parser.add_argument("--output", default=OUTPUT_VIDEO, help="output video path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video or 'webcam'}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (W, H)
    )

    tracker = OCSORTTracker(frame_rate=frame_rate)
    sam_model = SAM3Inference()
    pose_model = TVMInference()

    box_annot = sv.BoxAnnotator()
    label_annot = sv.LabelAnnotator()
    mask_annot = sv.MaskAnnotator()
    vertex_annot = sv.VertexAnnotator(color=sv.Color.RED, radius=3)

    frame_idx = 0
    with sv.JSONSink(JSON_PATH) as sink:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, masks = sam_model.predict(frame_rgb, text=args.text)

            if len(boxes):
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=np.ones(len(boxes), dtype=np.float32),
                    mask=masks,
                )
                tracked = tracker.update(detections)
                tracker_ids = tracked.tracker_id
            else:
                tracked = sv.Detections.empty()
                tracker_ids = np.zeros(0, dtype=int)

            # TVMInference uses cv2.dnn.blobFromImage(swapRB=True) → expects BGR.
            kpts = pose_model.predict(frame, tracked.xyxy)

            annotated = frame.copy()
            if len(tracked):
                annotated = mask_annot.annotate(annotated, tracked)
                annotated = box_annot.annotate(annotated, tracked)
                labels = [f"#{tid}" for tid in tracker_ids]
                annotated = label_annot.annotate(annotated, tracked, labels=labels)

            if kpts.size:
                # Zero out low-confidence keypoints so the annotator skips them.
                xy = kpts[:, :, :2].copy()
                conf = kpts[:, :, 2]
                xy[conf < KEYPOINT_THRESHOLD] = 0
                annotated = vertex_annot.annotate(
                    annotated, sv.KeyPoints(xy=xy, confidence=conf)
                )

            writer.write(annotated)

            for i in range(len(tracked)):
                sink.append(
                    sv.Detections(
                        xyxy=tracked.xyxy[i : i + 1],
                        tracker_id=tracker_ids[i : i + 1],
                    ),
                    custom_data={
                        "frame_idx": frame_idx,
                        "keypoints": kpts[i].tolist() if i < len(kpts) else [],
                    },
                )
            frame_idx += 1

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
