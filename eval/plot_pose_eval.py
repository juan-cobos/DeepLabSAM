"""Plot the CSVs written by scripts/eval_pose_coco.py.

Usage:
    uv run scripts/plot_pose_eval.py [--csv-prefix outputs/test_pose]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv-prefix", type=Path, default=Path("eval/outputs/test_pose"))
parser.add_argument("--out-dir", type=Path, default=Path("eval/outputs"))
parser.add_argument("--dpi", type=int, default=200)
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

metrics = {
    r["metric"]: r["value"]
    for _, r in pd.read_csv(
        args.csv_prefix.with_name(f"{args.csv_prefix.name}_metrics.csv")
    ).iterrows()
}
bodyparts = pd.read_csv(
    args.csv_prefix.with_name(f"{args.csv_prefix.name}_bodyparts.csv")
)
matches = pd.read_csv(args.csv_prefix.with_name(f"{args.csv_prefix.name}_matches.csv"))


# --- 1. mean OKS per bodypart ------------------------------------------------
scored = bodyparts.dropna(subset=["mean_oks"])
labels = (scored["bodypart"] + "  (" + scored["n_labeled"].astype(str) + ")").tolist()
vals = pd.to_numeric(scored["mean_oks"]).tolist()
n = len(labels)
n_unlabeled = len(bodyparts) - n
row_in = 0.26
figh = row_in * n + 0.72
fig, ax = plt.subplots(figsize=(1.45 + 4.0, figh))
ax.set_xlim(0, max(vals) * 1.16)
ax.set_ylim(n - 0.5, -0.5)
ax.set_yticks(range(n), labels)
ax.set_xticks([])
ax.tick_params(axis="y", length=0)
y1 = ax.get_position().y1
h = fig.get_figheight()
fig.text(
    ax.get_position().x0,
    y1 + 0.30 / h,
    "Mean OKS per bodypart",
    fontsize=10,
    weight="bold",
    va="baseline",
)
fig.text(
    ax.get_position().x0,
    y1 + 0.11 / h,
    f"Worst first · (n) = labeled ground truths · {n_unlabeled} joints unlabeled in this split",
    fontsize=7.5,
    color="gray",
    va="baseline",
)
bars = ax.barh(range(n), vals, height=0.5, color="#2a78d6")
pad = max(vals) * 0.01
for bar, v in zip(bars, vals):
    ax.text(
        v + pad,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.3f}",
        va="center",
        fontsize=7.5,
    )
fig.savefig(args.out_dir / "pose_bodyparts.png", dpi=args.dpi)
plt.close(fig)

# --- 2. COCO AP/AR -----------------------------------------------------------
ap_ar = [
    (k, float(v))
    for k, v in metrics.items()
    if k[:3] in ("AP_", "AR_") and "medium" not in k and "large" not in k
]
labels = [k for k, _ in ap_ar]
vals = [v for _, v in ap_ar]
n = len(labels)
figh = row_in * n + 0.72
fig, ax = plt.subplots(figsize=(1.0 + 4.0, figh))
ax.set_xlim(0, max(vals) * 1.16)
ax.set_ylim(n - 0.5, -0.5)
ax.set_yticks(range(n), labels)
ax.set_xticks([])
ax.tick_params(axis="y", length=0)
y1 = ax.get_position().y1
h = fig.get_figheight()
fig.text(
    ax.get_position().x0,
    y1 + 0.30 / h,
    "COCO keypoint AP / AR",
    fontsize=10,
    weight="bold",
    va="baseline",
)
fig.text(
    ax.get_position().x0,
    y1 + 0.11 / h,
    f"OKS thresholds · {metrics['n_images']} images, {metrics['n_ground_truths']} ground truths",
    fontsize=7.5,
    color="gray",
    va="baseline",
)
bars = ax.barh(range(n), vals, height=0.5, color="#2a78d6")
pad = max(vals) * 0.01
for bar, v in zip(bars, vals):
    ax.text(
        v + pad,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.3f}",
        va="center",
        fontsize=7.5,
    )
fig.savefig(args.out_dir / "pose_metrics.png", dpi=args.dpi)
plt.close(fig)

# --- 3. OKS distribution -----------------------------------------------------
oks = pd.to_numeric(matches.loc[matches["match"] == "tp", "oks"])
fig, ax = plt.subplots(figsize=(5.2, 3.1))
counts, _, _ = ax.hist(oks, bins=20, range=(0.5, 1.0), color="#2a78d6")
median = float(metrics["oks_median"])
ax.axvline(median, color="gray", linewidth=0.5)
ax.annotate(
    f"median {median:.3f}",
    (median, max(counts)),
    xytext=(-5, 0),
    textcoords="offset points",
    ha="right",
    va="top",
    fontsize=7.5,
)
ax.set_xlabel("OKS")
ax.set_ylabel("matched pairs")
ax.set_xlim(0.5, 1.0)
ax.grid(axis="y")
ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(True)
ax.tick_params(length=0)
y1 = ax.get_position().y1
h = fig.get_figheight()
fig.text(
    ax.get_position().x0,
    y1 + 0.30 / h,
    "OKS distribution over matched pairs",
    fontsize=10,
    weight="bold",
    va="baseline",
)
fig.text(
    ax.get_position().x0,
    y1 + 0.11 / h,
    f"{metrics['n_true_positives']} of {metrics['n_ground_truths']} ground truths "
    f"matched · {metrics['n_false_positives']} false positives · {metrics['n_missed']} missed",
    fontsize=7.5,
    color="gray",
    va="baseline",
)
fig.savefig(args.out_dir / "pose_oks_hist.png", dpi=args.dpi)
plt.close(fig)

# --- 4. OKS against ground-truth area ----------------------------------------
tp = matches[matches["match"] == "tp"]
fig, ax = plt.subplots(figsize=(5.2, 3.1))
ax.scatter(
    pd.to_numeric(tp["gt_area"]),
    pd.to_numeric(tp["oks"]),
    s=8,
    color="#2a78d6",
    edgecolors="white",
    linewidths=0.5,
)
ax.set_xscale("log")
ax.set_xlabel("ground-truth area (px²)")
ax.set_ylabel("OKS")
ax.grid(axis="y")
ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(True)
ax.tick_params(length=0)
y1 = ax.get_position().y1
h = fig.get_figheight()
fig.text(
    ax.get_position().x0,
    y1 + 0.30 / h,
    "Pose quality against animal size",
    fontsize=10,
    weight="bold",
    va="baseline",
)
fig.text(
    ax.get_position().x0,
    y1 + 0.11 / h,
    f"{len(tp)} matched pairs · one point per animal",
    fontsize=7.5,
    color="gray",
    va="baseline",
)
fig.savefig(args.out_dir / "pose_oks_vs_area.png", dpi=args.dpi)
plt.close(fig)
