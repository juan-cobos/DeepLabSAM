import numpy as np
import pytest

from deeplabsam.models.sam import OSAM


@pytest.fixture(scope="module")
def model():
    return OSAM(model="sam3:latest")


@pytest.mark.slow
def test_loads(model):
    assert model.model == "sam3:latest"


@pytest.mark.slow
def test_predict_with_box(model, example_image, example_box):
    H, W = example_image.shape[:2]
    boxes, masks = model.predict(example_image, text="mice", boxes=example_box)

    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert boxes.dtype == np.float32
    assert masks.dtype == bool
    assert masks.shape == (boxes.shape[0], H, W)
    assert boxes.shape[0] > 0, "expected at least one detection on example ROI"

    for b in boxes:
        assert 0 <= b[0] < b[2] <= W
        assert 0 <= b[1] < b[3] <= H

    # Masks must be non-empty and lie within their detection bbox.
    for b, m in zip(boxes.astype(int), masks):
        assert m.any()
        outside = m.copy()
        outside[b[1] : b[3] + 1, b[0] : b[2] + 1] = False
        assert not outside.any(), "mask pixels found outside the detection bbox"


@pytest.mark.slow
def test_predict_default_box_is_full_image(model, example_image):
    # box=None should fall back to the full-image box without errors.
    boxes, masks = model.predict(example_image, text="mice")
    H, W = example_image.shape[:2]
    assert masks.shape[1:] == (H, W)
    assert boxes.shape[1] == 4
