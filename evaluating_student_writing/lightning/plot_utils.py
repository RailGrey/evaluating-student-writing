import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _get_metric_history(
    client: MlflowClient, run_id: str, *metric_keys: str
) -> tuple[list[int], list[float]]:
    for key in metric_keys:
        try:
            history = client.get_metric_history(run_id, key)
            if history:
                steps = [m.step for m in history]
                values = [m.value for m in history]
                return steps, values
        except Exception:
            continue
    return [], []


def _generate_plots(
    client: MlflowClient, run_id: str, metrics_result: dict, plots_dir: Path
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_train_val_loss(client, run_id, plots_dir)
    _plot_learning_rate(client, run_id, plots_dir)
    _plot_metrics_over_val_steps(client, run_id, plots_dir)
    _plot_competition_f1_per_class(metrics_result, plots_dir)
    _plot_metrics_overview(metrics_result, plots_dir)

    for plot_file in sorted(plots_dir.glob("*.png")):
        mlflow.log_artifact(str(plot_file))
        logger.info("Logged plot artifact: %s", plot_file.name)


def _plot_train_val_loss(client: MlflowClient, run_id: str, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    train_steps, train_values = _get_metric_history(
        client, run_id, "train_loss_step", "train_loss"
    )
    val_steps, val_values = _get_metric_history(
        client, run_id, "val_loss_epoch", "val_loss"
    )

    if train_steps:
        ax.plot(train_steps, train_values, label="Train Loss", alpha=0.7)
    if val_steps:
        ax.plot(
            val_steps,
            val_values,
            label="Val Loss",
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = plots_dir / "train_val_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_rate(client: MlflowClient, run_id: str, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    steps, values = _get_metric_history(
        client, run_id, "learning_rate_step", "learning_rate"
    )
    if steps:
        ax.plot(steps, values, color="green")
        ax.fill_between(steps, values, alpha=0.15, color="green")

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    path = plots_dir / "learning_rate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_over_val_steps(
    client: MlflowClient, run_id: str, plots_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_keys = [
        ("competition_f1/macro_f1", "Competition F1"),
        ("token_f1/macro_f1", "Token F1"),
        ("span_exact_match_f1/macro_f1", "Span Exact Match"),
        ("span_jaccard_iou/macro_f1", "Span Jaccard"),
    ]
    colors = ["steelblue", "coral", "green", "orange"]
    has_data = False

    for (mlflow_key, label), color in zip(metric_keys, colors):
        steps, values = _get_metric_history(client, run_id, mlflow_key)
        if steps:
            ax.plot(steps, values, label=label, color=color, marker="o", markersize=4)
            has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Step")
    ax.set_ylabel("Macro F1")
    ax.set_title("Validation Metrics Over Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    path = plots_dir / "metrics_over_val_steps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_competition_f1_per_class(metrics_result: dict, plots_dir: Path) -> None:
    per_class = metrics_result.get("competition_f1", {}).get("per_class", {})
    if not per_class:
        return

    classes = sorted(per_class.keys())
    f1_scores = [per_class[c]["f1"] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(range(len(classes)))
    bars = ax.bar(classes, f1_scores, color=colors, edgecolor="gray", alpha=0.85)

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("F1 Score")
    ax.set_title("Competition F1 — Per Class")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

    path = plots_dir / "competition_f1_per_class.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_overview(metrics_result: dict, plots_dir: Path) -> None:
    metric_names = [
        "competition_f1",
        "token_f1",
        "span_exact_match_f1",
        "span_jaccard_iou",
    ]
    labels = ["Competition F1", "Token F1", "Span Exact Match", "Span Jaccard"]
    macro_f1s = []
    micro_f1s = []

    for m in metric_names:
        averages = metrics_result.get(m, {}).get("averages", {})
        macro_f1s.append(averages.get("macro", {}).get("f1", 0.0))
        micro_f1s.append(averages.get("micro", {}).get("f1", 0.0))

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        [i - width / 2 for i in x],
        macro_f1s,
        width,
        label="Macro",
        color="steelblue",
        alpha=0.85,
    )
    ax.bar(
        [i + width / 2 for i in x],
        micro_f1s,
        width,
        label="Micro",
        color="coral",
        alpha=0.85,
    )

    for i, (mac, mic) in enumerate(zip(macro_f1s, micro_f1s)):
        ax.text(
            i - width / 2,
            mac + 0.005,
            f"{mac:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            i + width / 2,
            mic + 0.005,
            f"{mic:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Metrics Overview — Macro / Micro Averages")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = plots_dir / "metrics_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
