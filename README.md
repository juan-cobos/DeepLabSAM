# DeepLabSAM

> Pose estimation with pixel-accurate masks.

<!-- Showcase: replace the placeholder below with annotated output images -->
<!-- Example layout:
| DeepLabCut | DeepLabSAM |
|:---:|:---:|
| ![dlc](docs/images/example_dlc.png) | ![deeplabsam](docs/images/example_deeplabsam.png) |
-->

---

## What it does

DeepLabSAM combines [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) ONNX pose models with [SAM3](https://github.com/wkentaro/osam) segmentation to produce keypoints **and** pixel-accurate instance masks in a single pass.

| Capability | DeepLabCut | DeepLabSAM |
|---|:---:|:---:|
| Keypoint estimation | ✓ | ✓ |
| Instance masks | — | ✓ |
| Text prompts | — | ✓ |
| `supervision` compatible | — | ✓ |

---

## Installation

Requires Python ≥ 3.13 and a running [osam](https://github.com/wkentaro/osam) server with a SAM model pulled.

```bash
# pull a SAM model (once)
osam pull sam3:latest

# run without installing
uvx deeplabsam --help

# or install into your environment
pip install deeplabsam
```

For GPU inference, install the CUDA-enabled ONNX runtime before installing the package:

```bash
pip install onnxruntime-gpu
pip install deeplabsam
```

---

## Usage

### CLI

```bash
# single image
deeplabsam image photo.png --text "mice" --output annotated.png

# video file
deeplabsam video recording.mp4 --text "mice" --output out.mp4 --json results.json

# webcam
deeplabsam video --text "mice"
```

**Common options**

| Flag | Default | Description |
|---|---|---|
| `--text` | `mice` | Text prompt passed to SAM |
| `--sam-model` | `sam3:latest` | osam model identifier |
| `--pose-model` | `topviewmouse` | DLC pose model from the registry |
| `--box XMIN YMIN XMAX YMAX` | — | Optional bounding box hint |
| `--keypoint-threshold` | `0.3` | Confidence cutoff for keypoint display |
| `--iou-threshold` | `0.5` | SAM IoU threshold |
| `--score-threshold` | `0.1` | SAM score threshold |
| `--output` | — | Output path (image or video) |
| `--json` | — | Keypoint + tracking annotations (video only) |

### Python API

```python
import cv2
from deeplabsam import DeepLabSAM

pipeline = DeepLabSAM(
    sam_model="sam3:latest",
    pose_model="topviewmouse",
    keypoint_threshold=0.3,
)

image = cv2.imread("photo.png")
detections, keypoints = pipeline.predict(image, text="mice")

# detections → sv.Detections  (xyxy boxes + pixel masks)
# keypoints  → sv.KeyPoints   (xy coords + confidence per keypoint)
```

---

## Available pose models

| Key | Species | Architecture |
|---|---|---|
| `topviewmouse` | Mouse (top view) | HRNet-W32 |

---

## Development

```bash
git clone https://github.com/JCobosAlvarez/DeepLabSAM
cd DeepLabSAM
uv sync
uv run pytest -m "not slow"
```
