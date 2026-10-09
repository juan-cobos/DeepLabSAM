# DeepLabSAM

![DeepLabSAM architecture: a video and a text prompt go into SAM 3 Video (detect · segment · track), whose masks gate the per-instance crop fed to a DeepLabCut pose head (keypoint regression), producing instance masks + tracks, a mask-gated instance crop, and tracked pose.](assets/architecture.png)

*DeepLabSAM converts raw multi-animal video into identity-resolved keypoints from a single text prompt.*

---

## What it does

DeepLabSAM is a unified, training-free behavioural analysis pipeline: it
streams a video through **[SAM 3](https://huggingface.co/facebook/sam3)** for
open-vocabulary detection, segmentation, and tracking, then hands each tracked
instance's **mask-gated crop** to a pretrained **DeepLabCut** pose head.

---

## Installation

```bash
pip install deeplabsam
```

For development, or to run the latest unreleased code, install from source
with [uv](https://docs.astral.sh/uv/) instead:

```bash
git clone https://github.com/juan-cobos/DeepLabSAM
cd DeepLabSAM
uv sync
```

`facebook/sam3` weights are gated — request access on the
[model page](https://huggingface.co/facebook/sam3), then set `HF_TOKEN` via
`.env` (loaded automatically) or the environment:

```bash
echo "HF_TOKEN=hf_..." > .env
```

Already have the weights locally? Point `--checkpoint-path`/`checkpoint_path=`
at that directory instead of the `facebook/sam3` repo id to skip Hugging Face
entirely.

---

## Usage

### CLI

`uv sync` installs a `deeplabsam` command with three subcommands. Run
`deeplabsam --help` or `deeplabsam <command> --help` for the full flag list
(`--help` is instant — the heavy model imports are deferred until a run
actually starts).

#### `deeplabsam run` — single-shot

Detect + track + pose in one pass, straight to an annotated video:

```bash
deeplabsam run path/to/video.mp4 --text mice
```

Every knob of the run is a flag — the SAM 3 build (`--image-size`,
`--score-threshold`, `--num-maskmem`), the DLC pose head (`--super-animal`,
`--pose-model`, `--pose-input-size`, `--device`) and the per-run options. Repeat
`--text`/`-t` for several prompts; animal prompts get pose, others are
mask/track only. With multiple prompts (e.g. `-t mouse -t object`), per-frame
class-agnostic NMS (`--class-agnostic`, `--nms-threshold`) drops duplicate
boxes when overlapping prompts match the same instance:

```bash
deeplabsam run path/to/video.mp4 \
    -t mouse -t object \
    --image-size 1008 \
    --max-frames 200 \
    --output-dir outputs \
    --keypoint-threshold 0.3 \
    --nms-threshold 0.5
```

#### `deeplabsam segment` + `deeplabsam pose` — curated workflow

For datasets you want to hand-curate before pose is fit (drop bad frames, fix
a track ID swap, …), split the run into two stages around a standard COCO
file:

```bash
# 1. Track the video, write frames/ + a COCO annotations.json for curation.
deeplabsam segment path/to/video.mp4 \
    --out-dir outputs/my_video \
    --prompt mouse \
    --max-detections 1

# 2. Curate outputs/my_video/annotations.json + frames/ by hand, then fit pose.
deeplabsam pose \
    --in outputs/my_video/annotations.json \
    --out outputs/my_video/annotations_pose.json
```

`segment` writes `<out-dir>/frames/*.jpg` (raw decoded frames), optionally
`<out-dir>/viz/*.png` (masks/boxes/labels drawn, for a quick sanity check
without opening an annotation tool), and `<out-dir>/annotations.json` (COCO
detections + segmentations, the SAM 3 score, and the persistent `track_id`).
Edit `annotations.json` — or drop frames — with any COCO-aware tool, then run
`pose`, which reads the (possibly curated) file, fits pose per instance
mask-gated to its segmentation, and writes the keypoints back in COCO
person-keypoints form (plus a `_check.mp4` QC video by default).

### Python API

```python
from deeplabsam.pipeline import Pipeline

# Build once — loads SAM 3 + the pose head onto the GPU.
# `Pipeline.default` wires a standard SAM 3 wrapper + DLC pose head for you.
pipe = Pipeline.default(super_animal="superanimal_topviewmouse")

# Run per video → writes an annotated mp4 (+ a sibling JSON) and returns its path.
out = pipe.run(
    video_path="recording.mp4",
    text="mice",  # open-vocabulary prompt; str or list[str]
    output_dir="outputs",  # <video-stem>_annotated.mp4 is written here
    max_frames=None,  # cap frames for a quick test
    keypoint_threshold=0.3,  # hide low-confidence keypoints
    export_json=True,  # also write <video-stem>_annotated.json
    save_masks=False,  # dump per-frame masks to outputs/<stem>_masks/*.npz
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

## Citation

If DeepLabSAM helps your research, please cite [the accompanying paper](https://www.biorxiv.org/content/10.64898/2026.10.02.756254v1):

```bibtex
@article{Cobos2026.10.02.756254,
  author    = {Cobos, Juan and Thirard, Steeve and Belkaid, Marwen and Naude, Jeremie},
  title     = {A promptable foundation model enables automated multi-task dataset construction and real-time pose estimation in mice},
  journal   = {bioRxiv},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.64898/2026.10.02.756254},
  URL       = {https://www.biorxiv.org/content/10.64898/2026.10.02.756254v1}
}
```

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
