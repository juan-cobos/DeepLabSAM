"""End-to-end torch pipeline as a class: SAM 3 video + DeepLabCut pose.

    SAM3Video.stream  ->  SegmentResult (boxes/masks/track-ids, GPU tensors)
                          |-> DLCTorchPose.predict_tensor (on-GPU crop -> keypoints)
                          |-> sv.Detections (masks/boxes) for annotation
    annotate (masks + boxes + track id + keypoints) -> mp4

Uses the HF (Transformers) SAM 3 video backend. The raw, pre-NMS boxes and the
on-device frame feed pose directly, so cropping stays on the GPU SAM 3 ran on;
NMS is not applied — the annotated video shows every raw detection.
"""

import cv2
import supervision as sv

from deeplabsam.pose.dlc import DLCTorchPose
from deeplabsam.segment.sam3video import SAM3Video


def rgb_frames(cap, max_frames=None):
    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        n += 1


class Pipeline:
    """SAM 3 video (HF, detect + track) + DeepLabCut pose, end to end.

    Build once (loads both models), then call :meth:`run` per video. ``device``
    and ``super_animal`` default to auto-CUDA and the top-view mouse head.
    """

    def __init__(self, device=None, super_animal="superanimal_topviewmouse"):
        self.predictor = SAM3Video(device=device)
        self.pose_head = DLCTorchPose(super_animal=super_animal, device=device)

    def run(
        self,
        video_path,
        text="mouse",
        max_frames=None,
        output_path="pipeline_out.mp4",
        keypoint_threshold=0.3,
    ):
        """Stream ``video_path``, detect+track+pose, write an annotated mp4.

        Returns ``output_path``.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")
        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (W, H)
        )
        mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        vertex_annot = sv.VertexAnnotator(color=sv.Color.RED, radius=3)

        for res in self.predictor.stream(rgb_frames(cap, max_frames), text):
            annotated = cv2.cvtColor(res.frame, cv2.COLOR_RGB2BGR)
            if not res:
                writer.write(annotated)
                continue

            # Raw, pre-NMS boxes + the on-GPU frame feed pose directly: the frame
            # is uploaded once and boxes already live on the inference device, so
            # the letterbox + pose run on the GPU SAM 3 ran on.
            kpts = self.pose_head.predict_tensor(res.frame_tensor(), res.boxes)
            xy = kpts[:, :, :2].copy()
            conf = kpts[:, :, 2]
            xy[conf < keypoint_threshold] = 0  # let annotator skip them
            keypoints = sv.KeyPoints(xy=xy, confidence=conf)

            det = res.to_detections()
            annotated = mask_annot.annotate(annotated, det)
            annotated = box_annot.annotate(annotated, det)
            labels = [f"#{tid}" for tid in det.tracker_id]
            annotated = label_annot.annotate(annotated, det, labels=labels)
            annotated = vertex_annot.annotate(annotated, keypoints)
            writer.write(annotated)

        cap.release()
        writer.release()
        print(f"Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    pipe = Pipeline()
    pipe.run(
        video_path="/home/juan/Videos/edit.mp4",
        text="mouse",
        max_frames=30,
        output_path="scripts/edit_pipeline_out.mp4",
    )
