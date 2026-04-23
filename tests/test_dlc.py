import numpy as np
import pytest

from deeplabsam.models.dlc import DLCPose, _REGISTRY


@pytest.fixture(scope="module", params=list(_REGISTRY))
def model(request):
    return DLCPose(model=request.param)


@pytest.mark.slow
def test_loads(model):
    assert model.session is not None
    inputs = {i.name for i in model.session.get_inputs()}
    outputs = {o.name for o in model.session.get_outputs()}
    assert "image" in inputs
    assert "poses" in outputs


@pytest.mark.slow
def test_empty_boxes(model, example_image):
    out = model.predict(example_image, np.zeros((0, 4)))
    assert out.shape == (0, 0, 3)


@pytest.mark.slow
def test_out_of_frame_box(model, example_image):
    out = model.predict(example_image, np.array([[-10, -10, -1, -1]]))
    assert out.shape == (1, 0, 3)
    assert np.all(out == 0)


@pytest.mark.slow
def test_inference_on_roi(model, example_image, example_box):
    H, W = example_image.shape[:2]
    boxes = np.array([example_box], dtype=np.float32)
    kpts = model.predict(example_image, boxes)

    assert kpts.ndim == 3
    assert kpts.shape[0] == 1
    assert kpts.shape[2] == 3
    assert kpts.shape[1] > 0

    xy, conf = kpts[0, :, :2], kpts[0, :, 2]
    assert (xy[:, 0] >= 0).all() and (xy[:, 0] <= W).all()
    assert (xy[:, 1] >= 0).all() and (xy[:, 1] <= H).all()
    assert conf.max() > 0.5


@pytest.mark.slow
def test_mixed_valid_invalid_boxes(model, example_image, example_box):
    boxes = np.array(
        [example_box, [-10, -10, -1, -1]],
        dtype=np.float32,
    )
    kpts = model.predict(example_image, boxes)
    assert kpts.shape[0] == 2
    assert np.all(kpts[1] == 0)
    assert kpts[0, :, 2].max() > 0.5
