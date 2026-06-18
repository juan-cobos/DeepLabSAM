import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from torch import nn
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

load_dotenv()
token = os.environ.get("HF_TOKEN")
login(token=token)


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
        return self.processor.postprocess_outputs(
            self.inference_session,
            model_outputs,
            original_sizes=inputs.original_sizes,  # Required for streaming inference
        )

    def _evict_old_memory(self, frame_idx: int) -> None:
        cutoff = frame_idx - self.memory_window
        if cutoff < 0:
            return
        # The session caches every input frame's pixel_values in ``processed_frames``
        # and never frees them. Forward streaming never re-reads frames older than the
        # window, so drop the aged-out ones — keeps the cache bounded (otherwise it
        # grows one frame per call: on the GPU by default, or host RAM when
        # ``video_storage_device`` is cpu).
        frames = self.inference_session.processed_frames
        if frames is not None:
            for k in list(frames):
                if k < cutoff:
                    del frames[k]
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
