"""Deterministic figures that reuse each representation's global embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_COLORS = {
    "left_hand": "#0072B2",
    "right_hand": "#E69F00",
    "feet": "#009E73",
    "tongue": "#D55E00",
}


def _category_colors(values: Iterable[object], category: str) -> dict[object, object]:
    unique = list(dict.fromkeys(values))
    if category == "class_label":
        return {value: CLASS_COLORS.get(str(value), "#666666") for value in unique}
    if category == "subject":
        cmap = plt.get_cmap("tab10")
        return {value: cmap(i % 10) for i, value in enumerate(sorted(unique))}
    if category == "window_index":
        cmap = plt.get_cmap("viridis", 5)
        return {value: cmap(int(value) - 1) for value in sorted(unique)}
    cmap = plt.get_cmap("tab10")
    return {value: cmap(i % 10) for i, value in enumerate(unique)}


def scatter_embedding(
    frame: pd.DataFrame,
    color_by: str,
    title: str,
    output_path: Path,
) -> None:
    colors = _category_colors(frame[color_by].tolist(), color_by)
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    for value in colors:
        group = frame[frame[color_by] == value]
        ax.scatter(
            group["tsne1"],
            group["tsne2"],
            s=8,
            alpha=0.55,
            linewidths=0,
            color=colors[value],
            label=str(value),
            rasterized=True,
        )
    ax.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(markerscale=2.2, fontsize=8, frameon=False, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def window_class_panels(
    frame: pd.DataFrame, title: str, output_path: Path
) -> None:
    """Five window panels using one already-fitted global t-SNE embedding."""
    x_limits = (float(frame.tsne1.min()), float(frame.tsne1.max()))
    y_limits = (float(frame.tsne2.min()), float(frame.tsne2.max()))
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharex=True, sharey=True)
    for window_index, ax in enumerate(axes, start=1):
        panel = frame[frame.window_index == window_index]
        for class_label, color in CLASS_COLORS.items():
            group = panel[panel.class_label == class_label]
            ax.scatter(
                group.tsne1,
                group.tsne2,
                s=7,
                alpha=0.55,
                linewidths=0,
                color=color,
                label=class_label,
                rasterized=True,
            )
        ax.set_title(f"Window {window_index}")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def example_trajectories(
    frame: pd.DataFrame,
    selected_subjects: list[int],
    title: str,
    output_path: Path,
) -> pd.DataFrame:
    """Plot the smallest trial ID per selected subject/class, deterministically."""
    selected_rows = []
    for subject in selected_subjects:
        subject_frame = frame[frame.subject == subject]
        for class_label in CLASS_COLORS:
            candidates = subject_frame[subject_frame.class_label == class_label]
            if candidates.empty:
                continue
            trial_id = candidates.trial_id.min()
            selected_rows.append(candidates[candidates.trial_id == trial_id])
    selected = pd.concat(selected_rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for (_, _, trial_uid), trial in selected.groupby(
        ["subject", "class_label", "trial_uid"], sort=True
    ):
        trial = trial.sort_values("window_index")
        class_label = str(trial.class_label.iloc[0])
        color = CLASS_COLORS[class_label]
        x = trial.tsne1.to_numpy()
        y = trial.tsne2.to_numpy()
        ax.plot(x, y, color=color, alpha=0.55, linewidth=1.0)
        ax.scatter(x, y, color=color, s=18, alpha=0.8, linewidths=0)
        for index in range(4):
            ax.annotate(
                "",
                xy=(x[index + 1], y[index + 1]),
                xytext=(x[index], y[index]),
                arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.45},
            )
    for class_label, color in CLASS_COLORS.items():
        ax.plot([], [], color=color, marker="o", label=class_label)
    ax.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(frameon=False, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return selected

