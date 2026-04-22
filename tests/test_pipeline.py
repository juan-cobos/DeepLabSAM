import pytest
import supervision as sv

from deeplabsam import DeepLabSAM


@pytest.fixture(scope="module")
def pipeline():
    return DeepLabSAM()


@pytest.mark.slow
def test_predict_on_roi(pipeline, example_image, example_box):
    detections, keypoints = pipeline.predict(
        example_image, text="mice", boxes=example_box
    )
    assert isinstance(detections, sv.Detections)
    assert isinstance(keypoints, sv.KeyPoints)
    assert len(detections) > 0
    assert detections.mask is not None and detections.mask.shape[0] == len(detections)
    # One set of keypoints per detection.
    assert keypoints.xy.shape[0] == len(detections)
    assert keypoints.xy.shape[2] == 2
    assert keypoints.confidence.max() > 0.5


@pytest.mark.slow
def test_no_detections_returns_empty(pipeline, example_image):
    # Score threshold above 1 filters out every detection.
    detections, keypoints = pipeline.predict(
        example_image, text="mice", score_threshold=2.0
    )
    assert len(detections) == 0
    assert keypoints.xy.size == 0
