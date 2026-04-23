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

```bash
# run without installing
uvx deeplabsam --help

# or install into your environment
uv add deeplabsam
```

For GPU inference, install the CUDA-enabled ONNX runtime before installing the package:

```bash
uv add onnxruntime-gpu
uv add "deeplabsam[gpu]"
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
| `quadruped` | Quadruped | HRNet-W32 |

---

## Available SAM models

Passed via `--sam-model` / `sam_model=` argument. Weights are downloaded automatically on first use.

| Key | Description |
|---|---|
| `sam3:latest` | SAM 3 (default, recommended) |
| `sam2:tiny` | SAM 2 Tiny |
| `sam2:small` | SAM 2 Small |
| `sam2:latest` | SAM 2 Base+ |
| `sam2:large` | SAM 2 Large |
| `sam:100m` | SAM ViT-B (100M) |
| `sam:300m` | SAM ViT-L (300M) |
| `sam:latest` | SAM ViT-H (600M) |
| `efficientsam:10m` | EfficientSAM 10M |
| `efficientsam:latest` | EfficientSAM 30M |
| `yoloworld:latest` | YOLO-World XL |

---

## Development

```bash
git clone https://github.com/JCobosAlvarez/DeepLabSAM
cd DeepLabSAM
uv sync
uv run pytest -m "not slow"
```
