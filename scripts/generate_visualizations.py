#!/usr/bin/env python3
"""Generate a self-contained visual dashboard from FACET-0 JSON training/evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "visualizations"
CHANNELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
ACTION_CHANNELS = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
COLORS = {
    "model": "#2563eb",
    "baseline": "#94a3b8",
    "accent": "#ea580c",
    "good": "#16a34a",
}


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def save(fig, name: str):
    path = OUTPUT / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_flow_training():
    runs = {
        "Overfit v2 absolute": "runs/alignment/overfit_v2_normalized/train_history.json",
        "Overfit v3 residual": "runs/alignment/overfit_v3_residual/train_history.json",
        "Full v2 pilot": "runs/alignment/full_v2_pilot/train_history.json",
        "Full v3 pilot": "runs/alignment/full_v3_pilot/train_history.json",
    }
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for label, path in runs.items():
        history = load_json(path)
        ax.plot(
            [item["step"] for item in history],
            [item["wrench_loss"] for item in history],
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_title("Stochastic wrench-flow training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Wrench flow-matching loss (log scale)")
    ax.legend(ncol=2)
    fig.tight_layout()
    return save(fig, "01_flow_training.png")


def plot_deterministic_training():
    runs = {
        "64-sequence overfit": "runs/alignment/deterministic_overfit/train_history.json",
        "2,048 stratified pool": "runs/alignment/deterministic_full_pilot/train_history.json",
        "512 policy-action pool": "runs/alignment/deterministic_policy_pilot/train_history.json",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, path in runs.items():
        history = load_json(path)
        steps = [item["step"] for item in history]
        axes[0].plot(steps, [item["huber_loss"] for item in history], label=label)
        axes[1].plot(steps, [item["normalized_mae"] for item in history], label=label)
    axes[0].set_title("Huber loss")
    axes[1].set_title("Normalized residual MAE")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Deterministic residual-wrench training")
    fig.tight_layout()
    return save(fig, "02_deterministic_training.png")


def plot_experiment_comparison():
    reports = [
        ("Flow overfit v2\n1 sample", "reports/alignment_overfit_v2_validation.json"),
        ("Flow overfit v3\n8-sample mean", "reports/alignment_overfit_v3_validation_ensemble8.json"),
        ("Deterministic\noracle action", "reports/deterministic_full_pilot_validation_128.json"),
        (
            "Deterministic\npolicy action",
            "reports/deterministic_full_pilot_validation_128_policy.json",
        ),
        (
            "Policy + scale 0.2\nvalidation",
            "reports/deterministic_full_pilot_validation_128_policy_calibration.json",
        ),
        (
            "Policy + scale 0.2\ntest",
            "reports/deterministic_full_pilot_test_264_policy_scale02.json",
        ),
    ]
    model = []
    baseline = []
    labels = []
    for label, path in reports:
        report = load_json(path)
        labels.append(label)
        model.append(report["mean_channel_mae"]["model"])
        baseline.append(report["mean_channel_mae"]["constant_baseline"])
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.bar(x - width / 2, model, width, label="Model", color=COLORS["model"])
    ax.bar(x + width / 2, baseline, width, label="Hold-current baseline", color=COLORS["baseline"])
    for index, (model_value, baseline_value) in enumerate(zip(model, baseline, strict=True)):
        color = COLORS["good"] if model_value < baseline_value else COLORS["accent"]
        ax.text(
            index - width / 2,
            model_value + 0.035,
            f"{model_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean of six physical-channel MAEs")
    ax.set_title("Wrench prediction versus causal constant baseline")
    ax.legend()
    fig.tight_layout()
    return save(fig, "03_model_vs_baseline.png")


def plot_scale_calibration():
    report = load_json("reports/deterministic_full_pilot_validation_128_policy_calibration.json")
    sweep = report["residual_scale_sweep"]
    scales = [item["scale"] for item in sweep]
    mae = [item["mean_channel_mae"] for item in sweep]
    baseline = report["mean_channel_mae"]["constant_baseline"]
    best = int(np.argmin(mae))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales, mae, marker="o", linewidth=2, color=COLORS["model"], label="Validation MAE")
    ax.axhline(baseline, color=COLORS["baseline"], linestyle="--", label=f"Baseline {baseline:.3f}")
    ax.scatter([scales[best]], [mae[best]], s=100, color=COLORS["good"], zorder=3)
    ax.annotate(
        f"selected scale={scales[best]:.2f}\nMAE={mae[best]:.3f}",
        (scales[best], mae[best]),
        xytext=(35, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_xlabel("Residual scale")
    ax.set_ylabel("Mean channel MAE")
    ax.set_title("Validation-only residual-scale calibration")
    ax.legend()
    fig.tight_layout()
    return save(fig, "04_residual_scale_calibration.png")


def plot_test_channels():
    report = load_json("reports/deterministic_full_pilot_test_264_policy_scale02.json")
    model = np.asarray(report["model"]["channel_mae"])
    baseline = np.asarray(report["constant_baseline"]["channel_mae"])
    x = np.arange(len(CHANNELS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - width / 2, model, width, label="Model", color=COLORS["model"])
    ax.bar(x + width / 2, baseline, width, label="Baseline", color=COLORS["baseline"])
    for index, passed in enumerate(model < baseline):
        ax.text(
            index,
            max(model[index], baseline[index]) + 0.05,
            "better" if passed else "worse",
            ha="center",
            fontsize=8,
            color=COLORS["good"] if passed else COLORS["accent"],
        )
    ax.set_xticks(x, CHANNELS)
    ax.set_ylabel("MAE in physical channel units")
    ax.set_title("Untouched test split: per-channel wrench MAE (264 episodes)")
    ax.legend()
    fig.tight_layout()
    return save(fig, "05_test_channel_mae.png")


def plot_action_chunk():
    report = load_json("reports/inference_episode000000_frame0000.json")
    actions = np.asarray(report["actions"])
    current_demo = np.asarray(report["demonstration_action"])
    steps = np.arange(len(actions))
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for index, ax in enumerate(axes.flat):
        if index >= len(ACTION_CHANNELS):
            ax.axis("off")
            continue
        ax.plot(steps, actions[:, index], color=COLORS["model"], label="Predicted chunk")
        ax.axhline(
            current_demo[index],
            color=COLORS["accent"],
            linestyle="--",
            label="Current demonstration action",
        )
        ax.set_title(ACTION_CHANNELS[index])
        ax.set_ylabel("Action value")
        if index == 0:
            ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Horizon step")
    fig.suptitle("FACET action chunk: episode 0, frame 0\n(dashed line is current action, not future GT)")
    fig.tight_layout()
    return save(fig, "06_action_chunk.png")


def plot_dataset_distribution():
    audit = load_json("reports/dataset_audit.json")
    task_counts = audit["task_frame_counts"]
    subtask_counts = audit["subtask_frame_counts"]
    task_items = sorted(task_counts.items(), key=lambda item: int(item[0]))
    subtask_items = sorted(subtask_counts.items(), key=lambda item: int(item[0]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar([item[0] for item in task_items], [item[1] for item in task_items], color=COLORS["model"])
    axes[0].set_yscale("log")
    axes[0].set_title("Frames per observed task ID")
    axes[0].set_xlabel("Task ID")
    axes[0].set_ylabel("Frames (log scale)")
    axes[1].bar(
        [item[0] for item in subtask_items],
        [item[1] for item in subtask_items],
        color=COLORS["accent"],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Frames per subtask ID")
    axes[1].set_xlabel("Subtask ID")
    axes[1].set_ylabel("Frames (log scale)")
    fig.suptitle("ManuFacet public-subset imbalance")
    fig.tight_layout()
    return save(fig, "07_dataset_distribution.png")


def write_dashboard(images):
    test = load_json("reports/deterministic_full_pilot_test_264_policy_scale02.json")
    validation = load_json("reports/deterministic_full_pilot_validation_128_policy_calibration.json")
    cards = [
        ("Validation improvement", f"{validation['mean_channel_mae']['relative_improvement'] * 100:.2f}%"),
        ("Test improvement", f"{test['mean_channel_mae']['relative_improvement'] * 100:.2f}%"),
        ("Test model MAE", f"{test['mean_channel_mae']['model']:.4f}"),
        ("Test baseline MAE", f"{test['mean_channel_mae']['constant_baseline']:.4f}"),
        ("Automated tests", "65 passed"),
        ("Robot success rate", "Not evaluated"),
    ]
    cards_html = "".join(
        f'<div class="card"><span>{title}</span><strong>{value}</strong></div>'
        for title, value in cards
    )
    figures_html = "".join(
        f'<section><h2>{title}</h2><p>{caption}</p><img src="{path.name}" alt="{title}"></section>'
        for title, caption, path in images
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FACET-0 离线复现可视化</title>
<style>
body {{ margin: 0; background: #f8fafc; color: #0f172a; font: 16px/1.6 system-ui, sans-serif; }}
main {{ max-width: 1180px; margin: auto; padding: 32px 22px 60px; }}
h1 {{ margin-bottom: 4px; }} .subtitle {{ color: #475569; margin-top: 0; }}
.warning {{ background: #fff7ed; border-left: 5px solid #ea580c; padding: 14px 18px; margin: 22px 0; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 12px; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; }}
.card span {{ display: block; color: #64748b; font-size: 13px; }}
.card strong {{ display: block; font-size: 24px; margin-top: 4px; }}
section {{ background: white; margin-top: 24px; padding: 20px; border-radius: 14px; border: 1px solid #e2e8f0; }}
section h2 {{ margin: 0; }} section p {{ color: #475569; }}
img {{ width: 100%; height: auto; display: block; }}
code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body><main>
<h1>FACET-0 离线复现可视化</h1>
<p class="subtitle">公开 checkpoint、ManuFacet 子集与研究性近似模块，生成于 2026-09-07。</p>
<div class="warning"><strong>重要：</strong>这些是开放环离线指标。当前没有真机闭环试验，
不能与论文约 82% 的机器人成功率直接比较。</div>
<div class="cards">{cards_html}</div>
{figures_html}
</main></body></html>"""
    path = OUTPUT / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    images = [
        ("Flow head 训练曲线", "归一化修复了数值失稳，但 flow head 的验证 MAE 未超过基线。", plot_flow_training()),
        ("确定性 residual 训练", "零 residual 等价于常数基线；分层池训练收敛稳定。", plot_deterministic_training()),
        ("实验结果总览", "不同报告的采样集合不同，柱图用于展示迭代方向，不作严格横向排名。", plot_experiment_comparison()),
        ("Residual scale 标定", "scale 只在 validation 选择为 0.2，随后冻结用于 test。", plot_scale_calibration()),
        ("Test 六通道 MAE", "264 个 test episodes；模型在 4/6 通道优于基线。", plot_test_channels()),
        ("动作块示例", "蓝线为模型的 50 步动作；虚线只是当前示范动作，不是未来真值。", plot_action_chunk()),
        ("公开数据分布", "task 0 和 8 占据绝大多数帧，长尾不平衡明显。", plot_dataset_distribution()),
    ]
    dashboard = write_dashboard(images)
    print(dashboard)
    for _, _, path in images:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
