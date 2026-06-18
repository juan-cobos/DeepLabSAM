"""Regression tests for the SAM 3 video VRAM leak fix.

The leak: the model stores each frame's ``maskmem_features``/``maskmem_pos_enc``
in ``session.output_dict_per_obj[obj]["(non_)cond_frame_outputs"][frame]`` on the
GPU and never frees them, so those dicts grow ~linearly and OOM the GPU on long
clips. ``Sam3VideoWrapper._evict_old_memory`` bounds them to ``memory_window``
frames.

These tests exercise the eviction logic directly against a fake session, so they
need no GPU, model weights, or video — they just lock in that memory stays
bounded. The optional ``slow`` test runs the real per-frame path when a CUDA GPU,
gated weights, and a test clip are available.
"""

import os
import types

import pytest

from deeplabsam.segment.sam3video import Sam3VideoWrapper


def fake_session(cond_frames, non_cond_frames, processed_frames=()):
    """A stand-in session with one object's frame-output dicts populated.

    Values are sentinels — the eviction logic only looks at the integer frame
    keys, so this avoids needing real tensors. ``processed_frames`` stands in for
    the session's per-frame input cache (defaults to empty).
    """
    return types.SimpleNamespace(
        output_dict_per_obj={
            0: {
                "cond_frame_outputs": {f: object() for f in cond_frames},
                "non_cond_frame_outputs": {f: object() for f in non_cond_frames},
            }
        },
        processed_frames={f: object() for f in processed_frames},
    )


def evict(window, session, frame_idx):
    """Call the (unbound) method without constructing the wrapper (no model load).

    ``_evict_old_memory`` reads ``self.memory_window`` and ``self.inference_session``,
    so a SimpleNamespace stands in for ``self``.
    """
    Sam3VideoWrapper._evict_old_memory(
        types.SimpleNamespace(memory_window=window, inference_session=session),
        frame_idx,
    )


def test_evicts_non_cond_outside_window():
    session = fake_session(cond_frames=[0], non_cond_frames=range(200))
    evict(window=64, session=session, frame_idx=199)  # cutoff = 135
    kept = session.output_dict_per_obj[0]["non_cond_frame_outputs"]
    assert min(kept) >= 135  # everything older than the window is gone
    assert 199 in kept and 134 not in kept


def test_keeps_initial_cond_anchor_but_evicts_old_reconditioned():
    # cond frame 0 is the initial anchor; 100 is a later reconditioned cond frame.
    session = fake_session(cond_frames=[0, 100], non_cond_frames=[199])
    evict(window=64, session=session, frame_idx=199)  # cutoff = 135
    cond = session.output_dict_per_obj[0]["cond_frame_outputs"]
    assert 0 in cond  # anchor survives even though 0 < cutoff (model needs it)
    assert 100 not in cond  # aged-out reconditioned cond frame is evicted


def test_evicts_old_processed_frames():
    # The per-frame input cache grows every call; aged-out frames must be dropped
    # too, or they OOM the GPU (the storage device defaults to the inference one).
    session = fake_session(cond_frames=[0], non_cond_frames=[199],
                           processed_frames=range(200))
    evict(window=64, session=session, frame_idx=199)  # cutoff = 135
    assert min(session.processed_frames) >= 135
    assert 199 in session.processed_frames and 134 not in session.processed_frames


def test_noop_before_window_filled():
    session = fake_session(cond_frames=[0], non_cond_frames=[0, 1, 2])
    evict(window=64, session=session, frame_idx=10)  # cutoff < 0 -> early return
    assert len(session.output_dict_per_obj[0]["non_cond_frame_outputs"]) == 3


def test_memory_stays_bounded_over_a_long_stream():
    """The actual OOM regression: simulate streaming and assert the per-object
    output dict never grows unbounded. Without eviction this reaches ~5000."""
    window = 64
    session = fake_session(cond_frames=[0], non_cond_frames=[])
    sam = types.SimpleNamespace(memory_window=window, inference_session=session)
    non_cond = session.output_dict_per_obj[0]["non_cond_frame_outputs"]
    for frame_idx in range(1, 5000):
        non_cond[frame_idx] = object()  # the model stores this frame's maskmem
        Sam3VideoWrapper._evict_old_memory(sam, frame_idx)
    assert len(non_cond) <= window + 1  # bounded, not ~5000


@pytest.mark.slow
def test_vram_stays_flat_when_streaming():
    """Real path: drive the wrapper per frame and assert VRAM doesn't climb.

    Opt-in — needs CUDA, gated SAM 3 weights, and a clip via DEEPLABSAM_TEST_VIDEO.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    video = os.environ.get("DEEPLABSAM_TEST_VIDEO")
    if not video or not os.path.exists(video):
        pytest.skip("set DEEPLABSAM_TEST_VIDEO to a clip to run the leak test")

    from torchcodec.decoders import VideoDecoder

    wrapper = Sam3VideoWrapper(text="mice", memory_window=64).eval()
    decoder = VideoDecoder(video, dimension_order="NHWC")

    early = late = None
    for i, frame in enumerate(decoder):
        if i >= 600:
            break
        wrapper(frame)
        if i == 200:
            early = torch.cuda.memory_allocated()
        if i == 599:
            late = torch.cuda.memory_allocated()

    assert early is not None and late is not None
    # Pre-fix this grew ~3.6 MiB/frame (~1.4 GB over 400 frames); with eviction the
    # window is full by frame 200, so growth should be a small fraction of that.
    growth_mib = (late - early) / 1024 / 1024
    assert growth_mib < 200, f"VRAM grew {growth_mib:.0f} MiB over 400 frames"
