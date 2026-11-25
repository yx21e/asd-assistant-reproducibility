from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def build_threshold_sweep_figure(threshold_payload: list[dict]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    palette = {
        "ink": "#272727",
        "grid": "#CFCECE",
        "blue": "#0F4D92",
        "blue_light": "#3775BA",
        "violet": "#9A4D8E",
        "violet_light": "#EA84DD",
    }
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9.2,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "axes.edgecolor": palette["ink"],
            "axes.labelcolor": palette["ink"],
            "xtick.color": palette["ink"],
            "ytick.color": palette["ink"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.1,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    thresholds = sorted({item["threshold"] for item in threshold_payload})
    sets = ["Task~2 curated", "Task~2 large", "Task~3 curated", "Task~3 large"]
    colors = {
        "Task~2 curated": palette["blue"],
        "Task~2 large": palette["blue_light"],
        "Task~3 curated": palette["violet"],
        "Task~3 large": palette["violet_light"],
    }
    markers = {"faithfulness_pass": "o", "ar_pass": "s", "cvr_pass": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.25), sharey=True)
    for ax, metric, title in zip(
        axes,
        ["faithfulness_pass", "ar_pass", "cvr_pass"],
        ["Faithfulness", "Answer relevance", "Citation verification"],
    ):
        for label in sets:
            values = [
                next(
                    item[metric]
                    for item in threshold_payload
                    if item["evaluation_set"] == label and item["threshold"] == threshold
                )
                for threshold in thresholds
            ]
            ax.plot(
                thresholds,
                values,
                marker=markers[metric],
                linewidth=1.9,
                markersize=4.8,
                color=colors[label],
                label=label.replace("~", ""),
            )
        ax.set_title(title, fontsize=9.4, weight="bold")
        ax.set_xlabel("Cutoff")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color=palette["grid"], linewidth=0.55, alpha=0.65)
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f"{threshold:.2f}" for threshold in thresholds], fontsize=8.0)
    axes[0].set_ylabel("Pass rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.045), fontsize=8.0)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0), pad=1.0)
    fig.savefig(FIG_DIR / "figure3_threshold_sweep.png", dpi=300, bbox_inches="tight", pad_inches=0.045)
    fig.savefig(FIG_DIR / "figure3_threshold_sweep.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(FIG_DIR / "figure3_threshold_sweep.svg", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

def build_memory_design_figure(memory_sweep: list[dict], task3_stack: list[dict]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    colors = {
        "ink": "#272727",
        "muted": "#767676",
        "grid": "#CFCECE",
        "task2": "#0F4D92",
        "task3": "#9A4D8E",
        "accent": "#B64342",
    }
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.edgecolor": colors["ink"],
            "axes.labelcolor": colors["ink"],
            "xtick.color": colors["ink"],
            "ytick.color": colors["ink"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.1,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    task2_order = [
        "Mixed evidence corpus",
        "Evidence-priority context",
        "Supporting excerpt first",
        "Excerpt with answer lead",
        "Evidence quality check",
        "Short biomedical prompt",
        "Final relevance scoring",
    ]
    task2_rows = [
        row
        for name in task2_order
        for row in memory_sweep
        if row["task"] == "Task~2" and row["source_label"] == name
    ]
    x = list(range(len(task2_rows)))
    labels = [
        "Mixed",
        "Gene-prior",
        "Excerpt",
        "Answer lead",
        "Ref. check",
        "Concise",
        "Final",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 2.85),
        gridspec_kw={"width_ratios": [1.06, 1.16], "wspace": 0.28},
        constrained_layout=True,
    )
    ax = axes[0]
    ax.plot(
        x,
        [row["faithfulness"] for row in task2_rows],
        color=colors["task2"],
        marker="o",
        markersize=4.2,
        linewidth=1.7,
        label="Faithfulness",
    )
    ax.plot(
        x,
        [row["ar"] for row in task2_rows],
        color=colors["accent"],
        marker="s",
        markersize=4.0,
        linewidth=1.7,
        label="AR",
    )
    ax.set_title("Task 2 answer form", fontsize=8.8, weight="bold", pad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.72, 1.00)
    ax.grid(axis="y", color=colors["grid"], linewidth=0.55, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    short = {
        "Final evidence setting": "Final",
        "Self-routed evidence": "Self",
        "General-topic evidence": "General",
        "General-topic evidence + graph expansion": "Graph",
        "General-topic evidence + second pass": "Second",
        "Combined evidence setting": "Combined",
        "General-topic evidence + adaptive summary": "Adaptive",
        "Combined evidence setting + adaptive summary": "Combined+adapt.",
    }
    rows = list(reversed(task3_stack))
    y = list(range(len(rows)))
    for pos in y:
        ax.axhline(pos, color=colors["grid"], linewidth=0.45, alpha=0.55, zorder=0)
    ax.scatter(
        [row["paper_faithfulness"] for row in rows],
        y,
        marker="o",
        color=colors["task2"],
        s=24,
        label="Faithfulness",
        zorder=3,
    )
    ax.scatter(
        [row["paper_ar"] for row in rows],
        y,
        marker="s",
        color=colors["accent"],
        s=24,
        label="AR",
        zorder=3,
    )
    ax.scatter(
        [row["real_topic_hit"] for row in rows],
        y,
        marker="D",
        color=colors["task3"],
        s=22,
        label="Topic hit",
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([short[row["variant"]] for row in rows])
    ax.set_title("Task 3 evidence setting", fontsize=8.8, weight="bold", pad=5)
    ax.set_xlabel("Score")
    ax.set_xlim(0.76, 0.99)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95])
    ax.grid(axis="x", color=colors["grid"], linewidth=0.55, alpha=0.75)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels_for_legend = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_for_legend,
        frameon=False,
        fontsize=7.0,
        handlelength=1.2,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.56, -0.025),
        columnspacing=1.3,
    )

    fig.savefig(FIG_DIR / "figure4_memory_ablation_tradeoff.png", dpi=260, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(FIG_DIR / "figure4_memory_ablation_tradeoff.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(FIG_DIR / "figure4_memory_ablation_tradeoff.svg", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

COLORS = {
    "ink": "#272727",
    "muted": "#767676",
    "grid": "#CFCECE",
    "task2": "#0F4D92",
    "task3": "#9A4D8E",
    "accent": "#B64342",
    "soft": "#F3F6FA",
    "low": "#EAF0F7",
    "high": "#0F4D92",
}

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.1,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )

def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.045)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

def prompt_family(query: str) -> str:
    q = query.strip().lower()
    if q.startswith("what"):
        return "what"
    if q.startswith("how"):
        return "how"
    if q.startswith("summarize"):
        return "summarize"
    return "other"

def task3_heatmap() -> None:
    rows = read_jsonl(TASK3_LARGE)
    topics = [
        ("clinical_features", "Clinical"),
        ("diagnosis_screening", "Dx/screen"),
        ("biomarkers", "Biomarkers"),
        ("epidemiology_population", "Epi/pop."),
        ("methods_study_design", "Methods"),
        ("risk_pathogenesis", "Risk/path."),
        ("general_autism", "General"),
    ]
    families = ["what", "how", "summarize"]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("topic")), prompt_family(str(row.get("query", ""))))].append(float(row["faithfulness"]))
    data = np.full((len(topics), len(families)), np.nan)
    counts = np.zeros_like(data)
    for i, (topic, _) in enumerate(topics):
        for j, family in enumerate(families):
            values = grouped[(topic, family)]
            if values:
                data[i, j] = float(np.mean(values))
                counts[i, j] = len(values)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "autism_teal",
        [COLORS["low"], "#DDF3DE", "#AADCA9", COLORS["task2"]],
    )
    fig, ax = plt.subplots(figsize=(3.2, 2.45))
    image = ax.imshow(data, vmin=0.90, vmax=0.97, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(families)), ["What", "How", "Summ."])
    ax.set_yticks(range(len(topics)), [label for _, label in topics])
    ax.tick_params(length=0)
    for i in range(len(topics)):
        for j in range(len(families)):
            if math.isnan(data[i, j]):
                label = "--"
                color = COLORS["muted"]
            else:
                label = f"{data[i, j]:.3f}\n({int(counts[i, j])})"
                color = "white" if data[i, j] > 0.945 else COLORS["ink"]
            ax.text(j, i, label, ha="center", va="center", fontsize=6.8, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.ax.tick_params(labelsize=6.2)
    save(fig, "figure3_task3_topic_prompt_heatmap")
