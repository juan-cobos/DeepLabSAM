"""Ablation of the mask-gating mechanism."""

import os

# Needed by deterministic cuBLAS; must be set before CUDA is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from deeplabsam.pose.backends.dlc import DLCPoseHead
from deeplabsam.pose.from_coco import decode_segmentation

SUPER_ANIMAL = "superanimal_topviewmouse"
KEYPOINT_THRESHOLD = 0.3
MAX_FRAMES = None  # cap per clip for a quick run
SEED = 0
OUT_DIR = Path("benchmarks/results")
OUT_CSV = OUT_DIR / "mask_gating_ablation_1008.csv"
STRATA_CSV = OUT_DIR / "mask_gating_ablation_1008_strata.csv"
ANN_ROOT = Path("outputs")  # <run>/annotations.json + frames/
CLIPS = {
    "mice_3": "udmt_bm_3",
    "mice_5": "udmt_bm_5",
    "mice_7": "udmt_bm_7",
    "mice_10": "udmt_bm_10",
}
OCC_BINS = [0.0, 0.02, 0.05, 0.10, 0.20, 1.01]  # occlusion strata (right-open)
MIN_AREA = 1000
FIELDS = [
    "clip",
    "n_mice",
    "frame",
    "track_id",
    "condition",
    "occlusion_fraction",
    "area",
    "n_confident_kp",
    "n_on_own_mask",
    "n_on_other_mask",
    "n_background",
]
CONDITIONS = ("ungated", "gated")  # baseline -> proposed


def seed_everything(seed: int) -> None:
    """Seed every RNG and force deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def classify(pts_xy: np.ndarray, own: np.ndarray, others: np.ndarray) -> tuple:
    """Count points (M, 2) falling on own mask / another mask / background."""
    if len(pts_xy) == 0:
        return 0, 0, 0
    H, W = own.shape
    xs = np.clip(pts_xy[:, 0].round().astype(int), 0, W - 1)
    ys = np.clip(pts_xy[:, 1].round().astype(int), 0, H - 1)
    on_own = own[ys, xs]
    on_other = others[ys, xs] & ~on_own
    n_own, n_other = int(on_own.sum()), int(on_other.sum())
    return n_own, n_other, len(pts_xy) - n_own - n_other


def run_clip(clip: str, head: DLCPoseHead, rows: list) -> None:
    clip_dir = ANN_ROOT / CLIPS[clip]
    coco = json.loads((clip_dir / "annotations.json").read_text())
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    images = sorted(coco["images"], key=lambda im: im["frame_index"])
    if MAX_FRAMES:
        images = images[:MAX_FRAMES]

    for image in images:
        f = image["frame_index"]
        # Sub-MIN_AREA masks (fragments, ghosts, empties) aren't animals: neither
        # posed nor counted as another animal's mask.
        dets = sorted(
            (a for a in by_image.get(image["id"], []) if a["area"] >= MIN_AREA),
            key=lambda a: a["track_id"],
        )
        if not dets:
            continue
        bgr = cv2.imread(str(clip_dir / image["file_name"]))
        if bgr is None:
            raise FileNotFoundError(clip_dir / image["file_name"])
        H, W = bgr.shape[:2]
        masks = np.stack([decode_segmentation(a["segmentation"], H, W) for a in dets])
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))  # BGR -> RGB
        # COCO bbox is xywh; the pose head crops from xyxy.
        xywh = np.array([a["bbox"] for a in dets], dtype=np.float32)
        boxes = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
        # Same boxes, mask gate on vs off — the whole ablation in two calls.
        poses_gated = head.predict_tensor(rgb, boxes, masks)
        poses_ungated = head.predict_tensor(rgb, boxes, None)

        for i, r in enumerate(dets):
            own = masks[i].astype(bool)
            others = np.zeros_like(own)
            for j in range(len(dets)):
                if j != i:
                    others |= masks[j].astype(bool)
            x1, y1, x2, y2 = (int(round(float(v))) for v in boxes[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(own.shape[1], x2), min(own.shape[0], y2)
            box_area = max((x2 - x1) * (y2 - y1), 1)
            occ_frac = float(others[y1:y2, x1:x2].sum()) / box_area

            poses = {"gated": poses_gated[i], "ungated": poses_ungated[i]}
            for condition in CONDITIONS:
                kp = poses[condition]  # (K, 3)
                valid = kp[:, 2] >= KEYPOINT_THRESHOLD
                n_own, n_other, n_bg = classify(kp[valid, :2], own, others)
                rows.append(
                    {
                        "clip": clip,
                        "n_mice": len(dets),
                        "frame": f,
                        "track_id": int(r["track_id"]),
                        "condition": condition,
                        "occlusion_fraction": round(occ_frac, 4),
                        "area": r["area"],
                        "n_confident_kp": int(valid.sum()),
                        "n_on_own_mask": n_own,
                        "n_on_other_mask": n_other,
                        "n_background": n_bg,
                    }
                )


def pooled_rates(rows: list) -> dict:
    """Pooled (cross-assign %, on-own %) per condition over a set of rows."""
    rates = {}
    for condition in CONDITIONS:
        rs = [r for r in rows if r["condition"] == condition]
        n = max(sum(r["n_confident_kp"] for r in rs), 1)
        rates[condition] = (
            100 * sum(r["n_on_other_mask"] for r in rs) / n,
            100 * sum(r["n_on_own_mask"] for r in rs) / n,
        )
    return rates


def write_strata(rows: list) -> None:
    """Pooled cross-assignment / on-own rates (%) per occlusion stratum and condition."""
    with STRATA_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["occlusion", "condition", "n", "cross_assign", "on_own"])
        for lo, hi in zip(OCC_BINS, OCC_BINS[1:]):
            rs = [r for r in rows if lo <= r["occlusion_fraction"] < hi]
            if not rs:
                continue
            n_animals = len(rs) // len(CONDITIONS)
            label = f"{lo * 100:g}-{min(hi, 1) * 100:g}%"  # [lo, hi), no comma
            for condition, (cross, own) in pooled_rates(rs).items():
                w.writerow([label, condition, n_animals, round(cross, 2), round(own, 2)])


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for the pose head.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    head = DLCPoseHead(super_animal=SUPER_ANIMAL)
    rows: list = []
    for clip in CLIPS:
        seed_everything(SEED)  # per clip, so each result is independent of order
        run_clip(clip, head, rows)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_strata(rows)
    print(f"wrote {len(rows)} rows -> {OUT_CSV}, strata -> {STRATA_CSV}")


if __name__ == "__main__":
    main()
