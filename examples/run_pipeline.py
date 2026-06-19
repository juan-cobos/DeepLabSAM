"""End-to-end example: SAM 3 video detect+track + DeepLabCut pose.

Usage:
    uv run examples/run_pipeline.py <video_path> [max_frames]
"""

import sys

from transformers import Sam3VideoConfig

from deeplabsam.pipeline import Pipeline
from deeplabsam.pose.dlc import DLCTorchPose
from deeplabsam.segment.sam3video import Sam3VideoWrapper

video_path = sys.argv[1]
# Optional second arg caps the number of frames; omit to process the whole video.
max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None

config = Sam3VideoConfig.from_pretrained("facebook/sam3")
config.image_size = 560
config.score_threshold_detection = 0.5
predictor = Sam3VideoWrapper(config=config)
pose_head = DLCTorchPose(super_animal="superanimal_topviewmouse")
pipe = Pipeline(predictor=predictor, pose_head=pose_head)

output_path = pipe.run(
    video_path=video_path,
    text=["mouse"],
    max_frames=max_frames,
    output_dir="outputs",
    export_json=True,
)
print(f"Done. Annotated video written to {output_path}")
