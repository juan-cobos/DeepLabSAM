import os

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import Sam3VideoModel, Sam3VideoProcessor

from deeplabsam.segment.segment_result import SegmentResult

load_dotenv()


class SAM3Video:
    """Streaming, multi-prompt video segmentation + tracking via SAM 3.

    Wraps ``Sam3VideoModel`` + ``Sam3VideoProcessor`` in the *streaming* regime:
    frames are fed one at a time (never the whole video at once) and the model
    carries per-object mask memory through time, so each object keeps a stable
    ID across frames without an external tracker.

    Accepts one or more text prompts; every detected object is tagged with the
    prompt that found it (its class), which downstream routes to the matching
    pose head. Non-overlapping constraints are applied *per prompt group*, so a
    "mouse" mask and a "bird" mask may overlap while two mice will not.

    Design choices for the simple, robust case:

    - **Streaming** (``init_video_session(video=None)`` + ``model(frame=...)``):
      handles arbitrarily long videos without loading them into memory.
    - **State on CPU** (``inference_state_device`` / ``video_storage_device``):
      VRAM stays flat (~2 GB) regardless of video length; only host RAM grows.
    - **bf16**: ~3x faster than fp32 and halves VRAM, with no visible quality loss
      here.

    No re-prompting: prompts are set once at the start of the stream.
    Weights (``facebook/sam3``) are gated; set ``HF_TOKEN`` (e.g. in ``.env``).
    """

    def __init__(self, model="facebook/sam3", device=None, dtype=torch.bfloat16):
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        # bf16 on CPU is poorly supported; fall back to fp32 off the GPU.
        self.dtype = dtype if device == "cuda" else torch.float32
        self.device = device

        token = os.environ.get("HF_TOKEN")
        kw = {"token": token} if token else {}
        self.model = (
            Sam3VideoModel.from_pretrained(model, dtype=self.dtype, **kw)
            .to(device)
            .eval()
        )
        self.processor = Sam3VideoProcessor.from_pretrained(model, **kw)

    def stream(self, frames, prompts):
        """Segment + track objects matching ``prompts`` across a frame stream.

        Args:
            frames: iterable of RGB ``(H, W, 3)`` uint8 arrays.
            prompts: a text prompt or list of prompts (set once, no re-prompting).
                Each prompt is a class; detections are tagged with their prompt.

        Yields:
            :class:`SegmentResult` per frame, with persistent ``object_ids`` and
            a ``class_name`` per detection.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        class_to_id = {p: i for i, p in enumerate(prompt_list)}

        session = self.processor.init_video_session(
            video=None,
            inference_device=self.device,
            inference_state_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        self.processor.add_text_prompt(session, prompt_list)

        for frame in frames:
            frame = np.asarray(frame)
            H, W = frame.shape[:2]
            pixel_values = (
                self.processor.video_processor(videos=[[frame]], return_tensors="pt")
                .pixel_values_videos[0, 0]
                .to(self.device, self.dtype)
            )

            with torch.no_grad():
                out = self.model(inference_session=session, frame=pixel_values)

            result = self.processor.postprocess_outputs(
                session, out, original_sizes=[[H, W]]
            )

            object_ids = result["object_ids"].detach().cpu().numpy().astype(int)
            # Invert {prompt: [obj_ids]} → per-object class, in object_ids order.
            obj_to_prompt = {
                oid: prompt
                for prompt, oids in result["prompt_to_obj_ids"].items()
                for oid in oids
            }
            class_names = [obj_to_prompt[int(oid)] for oid in object_ids]
            class_ids = np.array([class_to_id[c] for c in class_names], dtype=int)

            yield SegmentResult(
                frame_idx=out.frame_idx,
                frame=frame,
                boxes=result["boxes"],
                masks=result["masks"],
                object_ids=object_ids,
                scores=result["scores"],
                class_names=class_names,
                class_ids=class_ids,
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
            print(
                f"frame {res.frame_idx}: {len(res)} objs, "
                f"ids={res.object_ids.tolist()}, classes={res.class_names}"
            )
    cap.release()
    dt = time.perf_counter() - t0
    print(f"{n} frames in {dt:.2f}s = {n / dt:.2f} fps")
    if model.device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
