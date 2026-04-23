import numpy as np
import supervision as sv

from deeplabsam.models.dlc import DLCPose
from deeplabsam.models.sam import OSAM


class DeepLabSAM:
    """Text/box-prompted segmentation + pose estimation.

    Composes OSAM (detection + masks) with DLCPose (keypoints). Returns
    supervision primitives so downstream trackers and annotators plug in
    without any adaptation.
    """

    def __init__(
        self,
        sam_model="sam3:latest",
        pose_model="topviewmouse",
        keypoint_threshold=0.3,
    ):
        self.sam = OSAM(model=sam_model)
        self.pose = DLCPose(model=pose_model)
        self.keypoint_threshold = keypoint_threshold

    def detect(
        self,
        image,
        text,
        *,
        boxes=None,
        points=None,
        point_labels=None,
        iou_threshold=0.5,
        score_threshold=0.1,
        max_annotations=100,
    ):
        """Run SAM on a frame and wrap the result in ``sv.Detections``."""
        out_boxes, masks = self.sam.predict(
            image,
            text=text,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_annotations=max_annotations,
        )
        if len(out_boxes) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=out_boxes,
            confidence=np.ones(len(out_boxes), dtype=np.float32),
            mask=masks,
        )

    def estimate_pose(self, image, detections):
        """Run pose on ``detections.xyxy`` and return ``sv.KeyPoints``.

        Low-confidence keypoints are zeroed so annotators skip them.
        """
        if len(detections) == 0:
            return sv.KeyPoints.empty()
        kpts = self.pose.predict(image, detections.xyxy)
        if kpts.size == 0:
            return sv.KeyPoints.empty()
        xy = kpts[:, :, :2].copy()
        conf = kpts[:, :, 2]
        xy[conf < self.keypoint_threshold] = 0
        return sv.KeyPoints(xy=xy, confidence=conf)

    def predict(self, image, text, **prompt_kwargs):
        """Detect + estimate pose in one call.

        Returns:
            (``sv.Detections``, ``sv.KeyPoints``).
        """
        detections = self.detect(image, text, **prompt_kwargs)
        keypoints = self.estimate_pose(image, detections)
        return detections, keypoints
