import numpy as np
import torch
from PIL import Image

# Meta's official SAM 3 repo (https://github.com/facebookresearch/sam3).
# Sam3VideoPredictor is the single-process predictor: it builds the dense
# video-tracking model and exposes the dict-based handle_request /
# handle_stream_request session API used by the example notebooks.
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

from deeplabsam.segment.frame_result import FrameResult


class SAM3Video:
    """Multi-prompt video segmentation + tracking via Meta's SAM 3 repo.

    Same interface and ``FrameResult`` contract as the Transformers-backed
    ``SAM3Video``, but built on the *official* ``facebookresearch/sam3``
    ``Sam3VideoPredictor`` instead of ``Sam3VideoModel`` / ``Sam3VideoProcessor``.

    How Meta's video predictor differs from the HF streaming regime:

    - **Clip-at-once, not frame-at-a-time.** ``init_state`` loads the whole clip
      up front (``start_session(resource_path=...)``). ``resource_path`` accepts a
      *list of PIL images*, so we feed the in-memory frames directly — no temp
      files. ``stream`` still materializes its ``frames`` iterable.
    - **VRAM stays flat** via ``offload_video_to_cpu`` / ``offload_state_to_cpu``:
      the frame tensors and per-object state live on host RAM; only the active
      window is on the GPU. This is the analogue of the HF "state on CPU" path.
    - **One text concept per session.** ``add_prompt`` resets the session state
      ("since it's a semantic prompt, we start over"), so a session tracks a
      single prompt. For several prompts we re-prompt + re-propagate the buffered
      clip once per prompt, offsetting object IDs between passes and merging the
      per-frame results — mirroring the repo's own multi-prompt ``forward``.
    - **bf16 is built in.** The model runs under an internal
      ``autocast(bfloat16)``; there is no ``from_pretrained(dtype=...)`` knob.
    - **Native 1008 resolution** is fixed in the builder (the detector backbone
      is resolution-locked); it is not a tuning knob.

    Model outputs arrive as CPU numpy (``out_obj_ids``, ``out_probs``,
    ``out_boxes_xywh`` in *normalized* xywh, ``out_binary_masks`` bool N×H×W);
    :meth:`_build_result` converts boxes to pixel xyxy and moves masks/scores
    back onto the device so the pose stage keeps consuming GPU tensors.

    Requires CUDA (the predictor and frame loader are CUDA-oriented). Weights
    are gated and auto-downloaded; set ``HF_TOKEN`` (e.g. in ``.env``).

    ``version`` selects the checkpoint + predictor:

    - ``"sam3"`` (default): ``facebook/sam3`` via ``Sam3VideoPredictor``.
    - ``"sam3.1"``: ``facebook/sam3.1`` (the *multiplex* tracker) via
      ``build_sam3_multiplex_video_predictor`` → ``Sam3MultiplexVideoPredictor``.
      Both share the same ``handle_request`` / ``handle_stream_request`` session
      API, so ``stream`` is identical — only construction differs. ``use_fa3``
      enables FlashAttention 3 (faster, but needs the FA3 package installed); it
      defaults to off so sam3.1 runs out of the box.
    """

    def __init__(self, version="sam3", device=None, compile=False, use_fa3=False):
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "SAM3Video (Meta backend) requires CUDA: the predictor "
                "and its frame loader are CUDA-only."
            )
        self.device = device
        self.version = version

        # The builders resolve the (gated) checkpoint + BPE asset and call
        # .cuda().eval() internally. Both predictors expose the same dict API.
        if version == "sam3.1":
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            self.predictor = build_sam3_multiplex_video_predictor(
                async_loading_frames=False,
                use_fa3=use_fa3,
                compile=compile,
            )
        elif version == "sam3":
            self.predictor = Sam3VideoPredictor(
                async_loading_frames=False,
                video_loader_type="cv2",
                compile=compile,
            )
        else:
            raise ValueError(f"version must be 'sam3' or 'sam3.1', got {version!r}")
        self.model = self.predictor.model

    def stream(self, frames, prompts):
        """Segment + track objects matching ``prompts`` across a frame stream.

        Args:
            frames: iterable of RGB ``(H, W, 3)`` uint8 arrays. Fully materialized
                (Meta's predictor needs the whole clip to start a session).
            prompts: a text prompt or list of prompts (set once, no re-prompting).
                Each prompt is a class; detections are tagged with their prompt.

        Yields:
            :class:`FrameResult` per frame, in frame order, with persistent
            ``object_ids`` and a ``class_name`` per detection. The single-prompt
            path yields lazily as propagation runs; the multi-prompt path buffers
            (it must re-propagate per prompt) and yields once all passes finish.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        class_to_id = {p: i for i, p in enumerate(prompt_list)}

        np_frames = [np.asarray(f) for f in frames]
        if not np_frames:
            return
        H, W = np_frames[0].shape[:2]
        pil_frames = [Image.fromarray(f) for f in np_frames]

        session_id = self._start_session(pil_frames)
        try:
            if len(prompt_list) == 1:
                yield from self._stream_single(
                    session_id, prompt_list[0], np_frames, H, W, class_to_id
                )
            else:
                yield from self._stream_multi(
                    session_id, prompt_list, np_frames, H, W, class_to_id
                )
        finally:
            self.predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

    def _start_session(self, pil_frames):
        """Open a session, calling ``init_state`` directly with version-correct kwargs.

        We bypass the ``start_session`` request here because the base predictor
        unconditionally forwards ``offload_state_to_cpu`` to ``init_state``, but
        sam3.1's multiplex ``init_state`` doesn't accept that kwarg. Calling the
        model directly lets each version get only the offload knobs it supports,
        then we register the session exactly as ``start_session`` would so the
        rest of the dict API (add_prompt / propagate / close) works unchanged.
        """
        import time
        import uuid

        kw = dict(
            resource_path=pil_frames,
            offload_video_to_cpu=True,
            async_loading_frames=False,
        )
        if self.version == "sam3":
            kw["offload_state_to_cpu"] = True  # multiplex (sam3.1) lacks this knob
            kw["video_loader_type"] = "cv2"
        inference_state = self.predictor.model.init_state(**kw)
        session_id = str(uuid.uuid4())
        self.predictor._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return session_id

    def _propagate(self, session_id, prompt):
        """Add ``prompt`` on frame 0 and forward-propagate; yield (frame_idx, out).

        Resets first so switching prompts is correct on sam3.1 (whose
        ``add_prompt`` does *not* auto-reset, unlike sam3); harmless on sam3.
        """
        self.predictor.handle_request(dict(type="reset_session", session_id=session_id))
        self.predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
            )
        )
        for response in self.predictor.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=0,
            )
        ):
            yield response["frame_index"], response["outputs"]

    def _stream_single(self, session_id, prompt, np_frames, H, W, class_to_id):
        for frame_idx, out in self._propagate(session_id, prompt):
            n = len(out["out_obj_ids"])
            yield self._build_result(
                frame_idx,
                np_frames[frame_idx],
                H,
                W,
                out["out_obj_ids"],
                out["out_probs"],
                out["out_boxes_xywh"],
                out["out_binary_masks"],
                [prompt] * n,
                class_to_id,
            )

    def _stream_multi(self, session_id, prompt_list, np_frames, H, W, class_to_id):
        n_frames = len(np_frames)
        acc = {
            i: {"obj_ids": [], "probs": [], "boxes": [], "masks": [], "classes": []}
            for i in range(n_frames)
        }
        # Each prompt's pass restarts object IDs near 0 (_propagate resets the
        # session first), so offset them into disjoint ranges before merging.
        offset = 0
        for prompt in prompt_list:
            max_oid = -1
            for frame_idx, out in self._propagate(session_id, prompt):
                a = acc[frame_idx]
                for k in range(len(out["out_obj_ids"])):
                    oid = int(out["out_obj_ids"][k])
                    a["obj_ids"].append(oid + offset)
                    a["probs"].append(float(out["out_probs"][k]))
                    a["boxes"].append(out["out_boxes_xywh"][k])
                    a["masks"].append(out["out_binary_masks"][k])
                    a["classes"].append(prompt)
                    max_oid = max(max_oid, oid)
            offset += max_oid + 1

        for frame_idx in range(n_frames):
            a = acc[frame_idx]
            masks = (
                np.stack(a["masks"]) if a["masks"] else np.zeros((0, H, W), dtype=bool)
            )
            yield self._build_result(
                frame_idx,
                np_frames[frame_idx],
                H,
                W,
                np.asarray(a["obj_ids"], dtype=int),
                np.asarray(a["probs"], dtype=np.float32),
                np.asarray(a["boxes"], dtype=np.float32).reshape(-1, 4),
                masks,
                a["classes"],
                class_to_id,
            )

    def _build_result(
        self,
        frame_idx,
        frame,
        H,
        W,
        obj_ids,
        probs,
        boxes_xywh,
        masks,
        class_names,
        class_to_id,
    ):
        """Assemble a :class:`FrameResult` from the model's CPU-numpy outputs.

        Boxes (normalized top-left xywh) become pixel xyxy; masks and scores
        move back onto the device so the pose stage keeps GPU tensors.
        """
        boxes = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] * W
        xyxy[:, 1] = boxes[:, 1] * H
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2]) * W
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3]) * H

        masks = np.ascontiguousarray(np.asarray(masks, dtype=bool))
        if masks.size == 0:
            masks = masks.reshape(0, H, W)

        return FrameResult(
            frame_idx=int(frame_idx),
            frame=frame,
            boxes=torch.from_numpy(xyxy).to(self.device),
            masks=torch.from_numpy(masks).to(self.device),
            object_ids=np.asarray(obj_ids, dtype=int),
            scores=torch.from_numpy(np.asarray(probs, dtype=np.float32)).to(
                self.device
            ),
            class_names=list(class_names),
            class_ids=np.array([class_to_id[c] for c in class_names], dtype=int),
        )


if __name__ == "__main__":
    import time

    import cv2

    video_path = "/home/juan/Videos/edit.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    def frame_iter(cap, limit=20):
        for _ in range(limit):
            ok, f = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    model = SAM3Video()
    t0 = time.perf_counter()
    n = 0
    for res in model.stream(frame_iter(cap), prompts=["mice"]):
        n += 1
        if res.frame_idx < 2 or res.frame_idx == 19:
            det = res.to_detections()
            print(
                f"frame {res.frame_idx}: {len(res)} objs, ids={res.object_ids.tolist()}, "
                f"classes={res.class_names}, det.class_name={list(det.data.get('class_name', []))}"
            )
    cap.release()
    dt = time.perf_counter() - t0
    print(f"{n} frames in {dt:.2f}s = {n / dt:.2f} fps")
    if model.device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
