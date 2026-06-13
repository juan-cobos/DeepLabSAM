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
masks, tracks, and pose keypoints from a single text prompt — no per-video
annotation, no `deeplabcut` install, no ONNX.

It chains two models in one CUDA process:

1. **[SAM 3 video](https://huggingface.co/facebook/sam3)** (HF Transformers) —
   open-vocabulary detection **and** tracking. A text prompt like `"mice"`
   segments every matching animal and carries a stable track ID across frames,
   so identical, frequently-interacting animals don't swap IDs on contact.
2. **DeepLabCut SuperAnimal pose** — the genuine DeepLabCut `PoseModel` (HRNet /
   ResNet heatmap heads), vendored and trimmed under `deeplabsam/pose/` so it
   runs in pure torch with no `deeplabcut` dependency. Weights are pulled from
   the DeepLabCut Model Zoo on first use.

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
video frames ──▶ SAM3Video.stream ──▶ SegmentResult (boxes / masks / track-ids, GPU tensors)
                                          │
                                          ├─▶ DLCTorchPose.predict_tensor   (on-GPU crop, mask-gated → keypoints)
                                          └─▶ SegmentResult.to_detections   (sv.Detections for annotation / JSON)
                                                     │
                                          annotate (masks + boxes + track id + keypoints) ──▶ annotated .mp4 + .json
```

Frames stream one at a time (bf16), the SAM 3 memory state stays on CPU, and the
boxes/frame stay on the GPU SAM 3 ran on — so pose cropping never leaves the
device. VRAM stays roughly flat (~2 GB) regardless of clip length.

---

## Requirements

- **Python 3.13+** and an **NVIDIA GPU** (CUDA). CPU runs but is impractically slow.
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

There is no CLI yet (planned). Use the Python API:

```python
from deeplabsam.pipeline import Pipeline

# Build once — loads SAM 3 + the pose head onto the GPU.
pipe = Pipeline(super_animal="superanimal_topviewmouse")

# Run per video → writes an annotated mp4 (+ a sibling JSON) and returns its path.
out = pipe.run(
    video_path="recording.mp4",
    text="mice",            # open-vocabulary prompt; str or list[str]
    output_dir="outputs",   # <video-stem>_annotated.mp4 is written here
    max_frames=None,        # cap frames for a quick test
    keypoint_threshold=0.3, # hide low-confidence keypoints
    export_json=True,       # also write <video-stem>_annotated.json
)
print(out)  # outputs/recording_annotated.mp4
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
| `export_json` | `True` | Write per-frame detections to a sibling `.json`. |
| `profile` | `False` | Print a per-stage (detect / pose / io) timing breakdown. |

### Lower-level building blocks

Use the stages directly when you need the tensors rather than a rendered video:

```python
from deeplabsam.segment.sam3video import SAM3Video
from deeplabsam.pose.dlc import DLCTorchPose

sam = SAM3Video()                                  # HF SAM 3 video, bf16 streaming
pose = DLCTorchPose(super_animal="superanimal_topviewmouse")

for res in sam.stream(rgb_frames, prompts="mice"):  # res: SegmentResult
    kpts = pose.predict_tensor(res.frame_tensor(), res.boxes, res.masks)  # (N, K, 3)
    det = res.to_detections()                       # sv.Detections (boxes + masks + ids)
```

---

## Outputs

- **`<stem>_annotated.mp4`** — masks, boxes, track IDs (`#0`, `#1`, …), and pose
  keypoints drawn per frame.
- **`<stem>_annotated.json`** — one row per detection via `sv.JSONSink`, each
  tagged with its `frame_index`:

  ```json
  {"x_min": 176.0, "y_min": 451.0, "x_max": 314.0, "y_max": 493.0,
   "class_id": 0, "confidence": 0.91, "tracker_id": 0,
   "class_name": "mice", "frame_index": 0}
  ```

  Masks are intentionally not serialized (they'd bloat the file); keypoints are
  rendered into the mp4 rather than the JSON.

---

## Pose models

Pass via `Pipeline(super_animal=...)`. Weights download from the DeepLabCut Model
Zoo on first use.

| `super_animal` | Subject | Backbone |
|---|---|---|
| `superanimal_topviewmouse` | Mouse, top view (default) | HRNet-W32 |
| `superanimal_quadruped` | Quadruped (side view) | HRNet-W32 |
| `superanimal_bird` | Bird | ResNet-50 |

The pose input crop size is configurable (`DLCTorchPose(input_size=...)`, must be
÷ the backbone's padding divisor — 32 for HRNet).

---

## Performance notes

- **bf16 streaming** keeps VRAM ~flat (~2 GB) and runs comfortably on a 12 GB GPU.
- Throughput is dominated by the SAM 3 forward pass (~90%+ of runtime); pose and
  annotation are a small remainder. As a reference point, a 5-mice top-view clip
  (652×636) runs ~2.9 fps end to end on an RTX 5070. Throughput scales mainly
  with the number of tracked animals.
- The SAM 3 detector resolution is **fixed at 1008×1008** by its pretrained 2D
  rotary position embeddings, so lowering resolution is not a speed knob. bf16
  and frame subsampling are the available levers.
- `transformers[kernels]` is pinned so SAM 3's mask post-processing (NMS, hole
  filling, sprinkle removal) is enabled — cleaner masks, which in turn feed the
  mask-gated pose crops.
- Run with `profile=True` to see the per-stage detect / pose / io split.

---

## Project layout

```
deeplabsam/
├── pipeline.py            # Pipeline: end-to-end SAM 3 video + DLC pose → mp4 + json
├── segment/
│   └── sam3video.py       # SAM3Video.stream (HF SAM 3 detect + track) + SegmentResult
└── pose/
    ├── dlc.py             # DLCTorchPose — vendored DeepLabCut SuperAnimal heads
    ├── modelzoo_utils.py  # Model Zoo snapshot/config fetch
    ├── configs/           # bundled SuperAnimal + backbone configs
    └── models/            # vendored DLC PoseModel (backbones / heads / predictors)
```

---

## Development

```bash
git clone https://github.com/juan-cobos/DeepLabSAM
cd DeepLabSAM
uv sync
```

Requires `HF_TOKEN` for the gated `facebook/sam3` weights. Status: pre-1.0
(`0.1.0`) — the API may change, and a CLI is planned.
