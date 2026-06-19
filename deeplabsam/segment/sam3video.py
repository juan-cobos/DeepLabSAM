import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import supervision as sv
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from torch import nn
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

load_dotenv()
token = os.environ.get("HF_TOKEN")
login(token=token)


@dataclass
class SegmentResult:
    """One frame of SAM 3 video detect+track output, with prompt/class attached.

    Wraps the raw ``postprocess_outputs`` dict so the prompt that found each object
    survives to the detection layer instead of being re-derived downstream. All
    tensors are aligned to ``object_ids`` order.
    """

    object_ids: torch.Tensor  # (N,)
    scores: torch.Tensor  # (N,)
    boxes: torch.Tensor  # (N, 4) xyxy
    masks: torch.Tensor  # (N, H, W) bool
    prompt_to_obj_ids: dict[str, list[int]]  # prompt text -> object IDs it found

    @classmethod
    def from_outputs(cls, outputs: dict) -> "SegmentResult":
        return cls(
            object_ids=outputs["object_ids"],
            scores=outputs["scores"],
            boxes=outputs["boxes"],
            masks=outputs["masks"],
            prompt_to_obj_ids=outputs["prompt_to_obj_ids"],
        )

    @cached_property
    def prompts(self) -> list[str]:
        """Per-detection prompt text, aligned to ``object_ids`` order."""
        obj_to_prompt = {
            oid: p for p, oids in self.prompt_to_obj_ids.items() for oid in oids
        }
        return [obj_to_prompt[int(oid)] for oid in self.object_ids.tolist()]

    def to_detections(self) -> sv.Detections:
        """Numpy ``sv.Detections`` for annotation/JSON, carrying class as well.

        ``class_id`` is a stable index per unique prompt (sorted by name, so it's
        consistent across frames) and the prompt text rides along in
        ``data["class_name"]`` — which ``sv.JSONSink`` serializes for free. This is
        the only place the GPU tensors are moved to the CPU.
        """
        object_ids = self.object_ids.detach().cpu().numpy().astype(int)
        if len(object_ids) == 0:
            return sv.Detections.empty()
        class_index = {name: i for i, name in enumerate(sorted(self.prompt_to_obj_ids))}
        prompts = self.prompts
        return sv.Detections(
            xyxy=self.boxes.detach().cpu().float().numpy().astype(np.float32),
            mask=self.masks.detach().cpu().numpy().astype(bool),
            confidence=self.scores.detach().cpu().float().numpy().astype(np.float32),
            tracker_id=object_ids,
            class_id=np.array([class_index[p] for p in prompts], dtype=int),
            data={"class_name": np.array(prompts, dtype=object)},
        )


class Sam3VideoWrapper(nn.Module):
    def __init__(
        self,
        text: str | list[str] | None = None,
        dtype=torch.bfloat16,
        state_device="cpu",
        video_storage_device="cpu",
        memory_window: int = 64,
        image_size: int = 1008,
    ):
        super().__init__()

        if not torch.cuda.is_available():
            raise ValueError()

        self.device = "cuda"
        self.dtype = dtype
        self.state_device = state_device
        self.memory_window = memory_window
        self.image_size = image_size
        self.video_storage_device = video_storage_device

        # SAM 3's default 1008x1008 input upscales smaller clips and dominates
        # runtime (~quadratic in resolution). ``image_size`` propagates through the
        # nested detector/tracker configs; pretrained weights load fine at smaller
        # sizes (pos-encodings interpolate), so this is a direct speed/accuracy knob.
        config = Sam3VideoConfig.from_pretrained("facebook/sam3")
        config.image_size = image_size
        self.model = Sam3VideoModel.from_pretrained(
            "facebook/sam3", config=config, dtype=self.dtype
        ).to(self.device)
        # The processor is device/dtype-agnostic at load time; it takes the device
        # per call (see ``forward``), so no device_map here. Match its resize target
        # to the model input so frames aren't sent through at the wrong resolution.
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        size = {"height": image_size, "width": image_size}
        self.processor.image_processor.size = size
        self.processor.video_processor.size = size
        self.inference_session = None
        if text is not None:
            self.reset(text)

    def reset(self, text: str | list[str]) -> "Sam3VideoWrapper":
        """Start a fresh tracking session for a new video with these prompts.

        The model weights stay loaded; only the streaming state is rebuilt, so one
        wrapper can process many videos without reloading SAM 3. The session object
        is created once and then reset in place — ``reset_state`` clears tracking,
        prompts and cache, so re-prompting can't leak a previous video's prompts.
        """
        if self.inference_session is None:
            self.inference_session = self.processor.init_video_session(
                video=None,
                dtype=self.dtype,
                inference_device=self.device,
                inference_state_device=self.state_device,
                video_storage_device=self.video_storage_device,
            )
        else:
            self.inference_session.reset_state()
        self.processor.add_text_prompt(self.inference_session, text)
        return self

    @torch.inference_mode()
    def forward(self, images, reverse=False):
        inputs = self.processor(images=images, device=self.device, return_tensors="pt")
        # Process frame using streaming inference - pass the processed pixel_values
        model_outputs = self.model(
            inference_session=self.inference_session,
            frame=inputs.pixel_values[0],
            reverse=reverse,
        )
        # Evict using the model's own frame index — the same value keyed into the
        # per-object output dicts below — so the cutoff lines up with what's stored.
        self._evict_old_memory(model_outputs.frame_idx)
        # Post-process outputs with original_sizes for proper resolution handling
        outputs = self.processor.postprocess_outputs(
            self.inference_session,
            model_outputs,
            original_sizes=inputs.original_sizes,  # Required for streaming inference
        )
        return SegmentResult.from_outputs(outputs)

    def _evict_old_memory(self, frame_idx: int) -> None:
        cutoff = frame_idx - self.memory_window
        if cutoff < 0:
            return
        for obj in self.inference_session.output_dict_per_obj.values():
            non_cond = obj["non_cond_frame_outputs"]
            for k in list(non_cond):
                if k < cutoff:
                    del non_cond[k]
            # Conditioning frames anchor the memory; the initial one must stay
            # (the model errors if it's gone). Evict only later reconditioned
            # cond frames that have aged out of the window.
            if cond := obj["cond_frame_outputs"]:
                anchor = min(cond)
                for k in list(cond):
                    if k < cutoff and k != anchor:
                        del cond[k]
