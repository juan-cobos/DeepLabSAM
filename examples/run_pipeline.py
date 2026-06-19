"""End-to-end example: SAM 3 video detect+track + DeepLabCut pose.

Usage:
    uv run examples/run_pipeline.py <video_path>
"""

import sys
import time

import torch
from torchcodec.decoders import VideoDecoder
from transformers import Sam3VideoConfig

from deeplabsam.pipeline import Pipeline
from deeplabsam.pose.dlc import DLCTorchPose
from deeplabsam.segment.sam3video import Sam3VideoWrapper

video_path = sys.argv[1]
# Build the components explicitly. Model knobs live on the config: image_size 560
# trades accuracy for speed (original resolution = 1008); score_threshold_detection
# 0.65 (SAM 3 default 0.5) drops low-confidence detections.
config = Sam3VideoConfig.from_pretrained("facebook/sam3")
config.image_size = 560
config.score_threshold_detection = 0.5
predictor = Sam3VideoWrapper(config=config)
pose_head = DLCTorchPose(super_animal="superanimal_topviewmouse")
pipe = Pipeline(predictor=predictor, pose_head=pose_head)
text = ["mouse"]
# Frame count up front so we can turn elapsed time into FPS.
meta = VideoDecoder(video_path, device="cpu").metadata
num_frames = meta.num_frames or 0

# Start memory accounting from a clean slate so the peak reflects this run.
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

t0 = time.perf_counter()
pipe.run(
    video_path=video_path,
    text=text,
    max_frames=num_frames,
    output_dir="outputs",
    export_json=True,
)
t1 = time.perf_counter()

dt = t1 - t0
peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
fps = num_frames / dt if dt else 0.0

print(f"frames:         {num_frames}")
print(f"elapsed:        {dt:.2f} s")
print(f"throughput:     {fps:.2f} fps")
print(f"peak allocated: {peak_alloc:.2f} GiB")
print(f"peak reserved:  {peak_reserved:.2f} GiB")
