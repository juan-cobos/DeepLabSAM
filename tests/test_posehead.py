"""Contract tests for the pose-head abstraction.

These pin down ``deeplabsam.pose.PoseHead`` as the structural contract the
pipeline depends on: anything exposing ``num_bodyparts`` + ``predict_tensor`` is
a valid pose backend and is interchangeable in the pipeline. They need no model
weights, GPU, or video — they exercise the Protocol itself, that the DLC backend
exposes the required surface, and that the pipeline accepts any conforming head.
"""

import numpy as np

from deeplabsam.pose import PoseHead
from deeplabsam.pose.backends.dlc import DLCPoseHead


class FakePoseHead:
    """Minimal conforming pose head — returns zero keypoints, no model needed."""

    def __init__(self, num_bodyparts: int = 5):
        self.num_bodyparts = num_bodyparts

    def predict_tensor(self, images, boxes, masks=None) -> np.ndarray:
        return np.zeros((len(boxes), self.num_bodyparts, 3), dtype=np.float32)


def test_conforming_object_is_a_posehead():
    assert isinstance(FakePoseHead(), PoseHead)


def test_missing_predict_tensor_is_not_a_posehead():
    class NoMethod:
        num_bodyparts = 5

    assert not isinstance(NoMethod(), PoseHead)


def test_missing_num_bodyparts_is_not_a_posehead():
    class NoAttr:
        def predict_tensor(self, images, boxes, masks=None):
            return np.zeros((0, 0, 3))

    assert not isinstance(NoAttr(), PoseHead)


def test_dlc_backend_exposes_the_contract():
    # Surface check only — instantiating would download weights. The real
    # isinstance(head, PoseHead) on a live DLCPoseHead is covered by usage.
    assert callable(getattr(DLCPoseHead, "predict_tensor", None))
    assert callable(getattr(DLCPoseHead, "from_dlc_project", None))


def test_predict_tensor_returns_n_by_k_by_3():
    head = FakePoseHead(num_bodyparts=7)
    out = head.predict_tensor(images=None, boxes=np.zeros((3, 4)))
    assert out.shape == (3, head.num_bodyparts, 3)


def test_pipeline_accepts_any_posehead_via_di():
    # The point of the refactor: Pipeline is typed against the Protocol, so a
    # fake (predictor + pose_head) wires up with no concrete model classes.
    from deeplabsam.pipeline import Pipeline

    class FakePredictor:
        def eval(self):
            return self

    pipe = Pipeline(predictor=FakePredictor(), pose_head=FakePoseHead())
    assert isinstance(pipe.pose_head, PoseHead)
