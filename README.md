# DeepLabSAM

> Text-prompted multi-animal pose tracking with pixel-accurate masks.

<!-- Showcase: replace the placeholder below with annotated output images -->
<!-- Example layout:
| DeepLabCut | DeepLabSAM |
|:---:|:---:|
| ![dlc](docs/images/example_dlc.png) | ![deeplabsam](docs/images/example_deeplabsam.png) |
-->

---

## What it does

DeepLabSAM is an end-to-end **torch** pipeline that turns a video into per-animal
masks, tracks, and pose keypoints from one or more text prompts — no per-video
annotation required.

It chains two models in one CUDA process:

1. **[SAM 3 video](https://huggingface.co/facebook/sam3)** (HF Transformers) —
   open-vocabulary detection **and** tracking. A text prompt like `"mice"`
   segments every matching animal and carries a stable track ID across frames,
   so identical, frequently-interacting animals don't swap IDs on contact.
2. **DeepLabCut SuperAnimal pose** — the genuine DeepLabCut `PoseModel` (HRNet /
   ResNet heatmap heads) running in pure torch, with no `deeplabcut` dependency.
   Weights are pulled from the DeepLabCut Model Zoo on first use, and you can also
   load your own DLC-trained model (a project config + snapshot) instead of a
   SuperAnimal one. The pose stage sits behind a small `PoseHead` contract, so the
   DLC backend can be swapped for another later.

The key trick: each animal's **mask gates its pose crop**. Instead of feeding the
raw bounding box (which, when two animals overlap, includes the neighbour's
body), the background is zeroed out so the pose head only ever sees the target
animal — cleaner keypoints under occlusion.

| Capability | DeepLabCut | DeepLabSAM |
|---|:---:|:---:|
| Keypoint estimation | ✓ | ✓ |
| Instance masks | — | ✓ |
| Multi-animal tracking (stable IDs) | — | ✓ |
| Text prompt (open vocabulary) | — | ✓ |
| Mask-gated pose crops | — | ✓ |
| `supervision`-native output | — | ✓ |

---

## How it works

```
video frames ──▶ Sam3VideoWrapper(frame) ──▶ SegmentResult (boxes / masks / track-ids, GPU tensors)
                                                │
                                                ├─▶ PoseHead.predict_tensor      (crop, mask-gated → keypoints)
                                                └─▶ SegmentResult.to_detections  (sv.Detections for annotation / JSON)
                                                           │
                                                annotate (masks + boxes + track id + keypoints) ──▶ annotated .mp4 + .json
```

With multiple prompts (e.g. `["mouse", "object"]`) only animal prompts are run
through pose; objects are masked/tracked only. Per-frame class-agnostic NMS drops
duplicate boxes when overlapping prompts match the same instance.

---

## Requirements

- **Python 3.11+** and an **NVIDIA GPU** (CUDA).
- A **Hugging Face token** — `facebook/sam3` weights are gated. Request access on
  the model page, then make the token available (see below). DeepLabCut Model Zoo
  pose weights download without a token.

---

## Installation

Not yet published to PyPI — install from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/juan-cobos/DeepLabSAM
cd DeepLabSAM
uv sync
```

Provide your Hugging Face token via a `.env` file (loaded automatically) or the
environment:

```bash
echo "HF_TOKEN=hf_..." > .env
# or: export HF_TOKEN=hf_...
```

---

## Usage

### CLI

`uv sync` installs a `deeplabsam` command that runs a video end to end:

```bash
deeplabsam path/to/video.mp4 --text mice
```

Every knob of the run is a flag — the SAM 3 build (`--image-size`,
`--score-threshold`, `--memory-window`), the DLC pose head (`--super-animal`,
`--pose-model`, `--pose-input-size`, `--device`) and the per-run options. Repeat
`--text`/`-t` for several prompts; animal prompts get pose, others are
mask/track only:

```bash
deeplabsam path/to/video.mp4 \
    -t mouse -t object \
    --image-size 1008 \
    --max-frames 200 \
    --output-dir outputs \
    --keypoint-threshold 0.3 \
    --nms-threshold 0.5
```

Run `deeplabsam --help` for the full flag list (`--help` is instant — the
heavy model imports are deferred until a run actually starts).

### Python API

```python
from deeplabsam.pipeline import Pipeline

# Build once — loads SAM 3 + the pose head onto the GPU.
# `Pipeline.default` wires a standard SAM 3 wrapper + DLC pose head for you.
pipe = Pipeline.default(super_animal="superanimal_topviewmouse")

# Run per video → writes an annotated mp4 (+ a sibling JSON) and returns its path.
out = pipe.run(
    video_path="recording.mp4",
    text="mice",            # open-vocabulary prompt; str or list[str]
    output_dir="outputs",   # <video-stem>_annotated.mp4 is written here
    max_frames=None,        # cap frames for a quick test
    keypoint_threshold=0.3, # hide low-confidence keypoints
    export_json=True,       # also write <video-stem>_annotated.json
    save_masks=False,       # dump per-frame masks to outputs/<stem>_masks/*.npz
)
print(out)  # outputs/recording_annotated.mp4
```

For full control — a custom SAM 3 config, a swapped pose backend, or test fakes —
build the components yourself and inject them (this is what `Pipeline.default`
and the CLI do under the hood):

```python
from transformers import Sam3VideoConfig
from deeplabsam.pipeline import Pipeline
from deeplabsam.pose.backends.dlc import DLCPoseHead
from deeplabsam.segment.sam3video import Sam3VideoWrapper

config = Sam3VideoConfig.from_pretrained("facebook/sam3")
config.image_size = 1008
pipe = Pipeline(
    predictor=Sam3VideoWrapper(config=config),
    pose_head=DLCPoseHead(super_animal="superanimal_topviewmouse"),
)
```

### `Pipeline.run` options

| Argument | Default | Description |
|---|---|---|
| `video_path` | — | Input video file. |
| `text` | `"mouse"` | Text prompt(s) for SAM 3 (`str` or `list[str]`). |
| `max_frames` | `None` | Process at most this many frames (`None` = whole video). |
| `output_dir` | `"outputs"` | Created if missing; output is named from the video stem. |
| `name_suffix` | `"_annotated"` | Appended to the stem (`<stem><name_suffix>.mp4`). |
| `keypoint_threshold` | `0.3` | Confidence cutoff below which keypoints are hidden. |
| `nms_threshold` | `0.5` | Per-frame box-IoU threshold for dropping duplicate detections. |
| `class_agnostic` | `False` | Run NMS across classes (drop overlaps regardless of prompt). |
| `export_json` | `True` | Write per-frame detections to a sibling `.json`. |
| `save_masks` | `False` | Dump each frame's masks as compressed `.npz` into a per-video `<output_dir>/<stem>_masks` folder. |

---

## Outputs

- **`<stem>_annotated.mp4`** — masks, boxes, track IDs (`#0`, `#1`, …), and pose
  keypoints drawn per frame.
- **`<stem>_annotated.json`** — one row per detection via `sv.JSONSink`, each
  tagged with its `frame_index` and per-instance `keypoints` (`[[x, y, conf], …]`,
  one entry per bodypart):

  ```json
  {"x_min": 176.0, "y_min": 451.0, "x_max": 314.0, "y_max": 493.0,
   "class_id": 0, "confidence": 0.91, "tracker_id": 0,
   "class_name": "<prompt>", "frame_index": 0,
   "keypoints": [[245.1, 470.3, 0.94], [251.7, 462.0, 0.88]]}
  ```

  `class_name` is the text prompt that matched the detection (e.g. `"mice"`), and
  `class_id` is its stable index across the prompt set.

- **`<stem>_masks/frame_<idx>.npz`** *(with `--save-masks` / `save_masks=True`)*
  — each frame's masks as a compressed `(N, H, W)` bool array under the `masks`
  key (`np.load(path)["masks"]`), aligned to detection order, in a per-video
  folder named after the input video so runs into the same `output_dir` don't
  overwrite each other.

---

## Pose models

Select via `--super-animal` on the CLI, or `Pipeline.default(super_animal=...)` /
`DLCPoseHead(super_animal=...)` in Python. Weights download from the DeepLabCut
Model Zoo on first use.

| `super_animal` | Subject | Backbone |
|---|---|---|
| `superanimal_topviewmouse` | Mouse, top view (default) | HRNet-W32 |
| `superanimal_quadruped` | Quadruped (side view) | HRNet-W32 |
| `superanimal_bird` | Bird | ResNet-50 |

The pose input crop size is configurable (`--pose-input-size`, or
`DLCPoseHead(input_size=...)`, default `256`; must be ÷ the backbone's padding
divisor — 32 for HRNet). To run a community DLC model beyond the SuperAnimal zoo,
build the head from a project config + snapshot with
`DLCPoseHead.from_dlc_project(config, snapshot_path)`.

---

## Development

```bash
git clone https://github.com/juan-cobos/DeepLabSAM
cd DeepLabSAM
uv sync
```

Requires `HF_TOKEN` for the gated `facebook/sam3` weights.

---

## Acknowledgements

DeepLabSAM stands on other projects — please cite/credit them if you use it:

- **[SAM 3](https://huggingface.co/facebook/sam3)** (Meta AI) — the open-vocabulary
  video detection and tracking model behind every mask and track ID.
- **[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)** (Mathis Lab) — the
  SuperAnimal `PoseModel` and Model Zoo weights that produce the keypoints; the
  pose backend vendors and runs their PyTorch model code.
- **[supervision](https://github.com/roboflow/supervision)** (Roboflow) — the
  detection/annotation/serialization layer (`sv.Detections`, the annotators, and
  `sv.JSONSink`).
- **[Hugging Face](https://github.com/huggingface/transformers)** — `transformers`
  serves the SAM 3 model/processor and `huggingface-hub` distributes the gated
  weights.
