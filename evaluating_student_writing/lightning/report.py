import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

METRIC_COLORS = {
    "competition_f1": "#e74c3c",
    "token_f1": "#2ecc71",
    "span_exact_match_f1": "#3498db",
    "span_jaccard_iou": "#e67e22",
}

METRIC_LABELS = {
    "competition_f1": "Competition F1",
    "token_f1": "Token F1",
    "span_exact_match_f1": "Span Exact Match",
    "span_jaccard_iou": "Span Jaccard",
}

DISCOURSE_COLORS = {
    "Lead": "#e74c3c",
    "Position": "#e67e22",
    "Claim": "#f1c40f",
    "Counterclaim": "#9b59b6",
    "Rebuttal": "#3498db",
    "Evidence": "#2ecc71",
    "Concluding Statement": "#1abc9c",
}


def generate_report(
    texts: dict[str, str],
    submission: pd.DataFrame,
    output_dir: Path,
    metrics_result: dict | None = None,
    filename: str = "inference_report.md",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename

    lines: list[str] = []
    lines.append("# Inference Report\n")
    lines.append(_prediction_stats(submission))
    lines.append(_color_legend())

    if metrics_result:
        lines.append(_macro_micro_summary(metrics_result))
        lines.append(_per_class_tables(metrics_result))

    lines.append(_essay_breakdown(texts, submission))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    return report_path


def _prediction_stats(submission: pd.DataFrame) -> str:
    parts = [
        f"**Essays:** {submission['id'].nunique()}  ",
        f"**Predicted spans:** {len(submission)}  ",
        "",
        "---",
        "",
    ]
    return "\n".join(parts)


def _color_legend() -> str:
    lines = ["## Discourse Type Colors", "", "<table>"]
    for cls, color in DISCOURSE_COLORS.items():
        lines.append(
            f'  <tr><td style="background:{color};color:white;padding:3px 12px;'
            f'border-radius:3px">{cls}</td></tr>'
        )
    lines.append("</table>\n")
    return "\n".join(lines)


def _macro_micro_summary(metrics_result: dict) -> str:
    lines = ["## Metrics — Macro / Micro Averages", ""]
    for avg in ["macro", "micro"]:
        lines.append(f"### {avg.capitalize()}")
        for key in METRIC_LABELS:
            color = METRIC_COLORS[key]
            label = METRIC_LABELS[key]
            vals = metrics_result.get(key, {}).get("averages", {}).get(avg, {})
            lines.append(
                f'<p><span style="color:{color};font-weight:bold">{label}</span>'
                f"&nbsp;&nbsp;F1: <b>{vals.get('f1', 0):.4f}</b>"
                f"&nbsp;&nbsp;P: {vals.get('precision', 0):.4f}"
                f"&nbsp;&nbsp;R: {vals.get('recall', 0):.4f}</p>"
            )
        lines.append("")
    return "\n".join(lines)


def _f1_bg(f1: float) -> str:
    if f1 >= 0.8:
        return "#27ae60"
    if f1 >= 0.6:
        return "#f39c12"
    if f1 >= 0.4:
        return "#e67e22"
    return "#e74c3c"


def _per_class_tables(metrics_result: dict) -> str:
    lines = ["## Per-Class Metrics", ""]
    for key in METRIC_LABELS:
        per_class = metrics_result.get(key, {}).get("per_class", {})
        if not per_class:
            continue
        color = METRIC_COLORS[key]
        label = METRIC_LABELS[key]
        lines.append(f'### <span style="color:{color}">{label}</span>\n')
        lines.append("<table>")
        lines.append(
            "<tr><th>Class</th><th>F1</th><th>Precision</th><th>Recall</th><th>Support</th></tr>"
        )
        for cls in sorted(per_class):
            d = per_class[cls]
            f1 = d["f1"]
            bg = _f1_bg(f1)
            tc = "white" if f1 < 0.6 else "#2c3e50"
            disc_color = DISCOURSE_COLORS.get(cls, "#95a5a6")
            lines.append(
                f"<tr>"
                f'<td style="color:{disc_color};font-weight:bold">{cls}</td>'
                f'<td style="background:{bg};color:{tc};padding:2px 8px;border-radius:3px;'
                f'font-weight:bold;text-align:center">{f1:.4f}</td>'
                f"<td>{d['precision']:.4f}</td>"
                f"<td>{d['recall']:.4f}</td>"
                f"<td>{d['support']}</td>"
                f"</tr>"
            )
        lines.append("</table>\n")
    return "\n".join(lines)


def _essay_breakdown(texts: dict[str, str], submission: pd.DataFrame) -> str:
    lines = ["## Essay Breakdown", ""]
    for essay_id in sorted(submission["id"].unique()):
        rows = submission[submission["id"] == essay_id]
        text = texts.get(essay_id, "")
        words = text.split() if text else []

        lines.append(f"### `{essay_id}` ({len(words)} words)\n")

        span_indices: dict[int, str] = {}
        for _, row in rows.iterrows():
            cls = row["class"]
            for idx_str in str(row["predictionstring"]).split():
                span_indices[int(idx_str)] = cls

        segments: list[str] = []
        prev_cls = None
        prev_words: list[str] = []
        prev_indices: list[int] = []

        for i, word in enumerate(words):
            cls = span_indices.get(i, "O")
            if cls != prev_cls:
                if prev_cls is not None:
                    segments.append(_render_segment(prev_cls, prev_words, prev_indices))
                prev_cls = cls
                prev_words = [word]
                prev_indices = [i]
            else:
                prev_words.append(word)
                prev_indices.append(i)
        if prev_cls is not None:
            segments.append(_render_segment(prev_cls, prev_words, prev_indices))

        lines.append('<p style="line-height:1.8">' + " ".join(segments) + "</p>")
        lines.append("")

    return "\n".join(lines)


def _render_segment(cls: str, words: list[str], indices: list[int]) -> str:
    text = " ".join(words)
    if cls == "O":
        return text
    color = DISCOURSE_COLORS.get(cls, "#95a5a6")
    return (
        f'<span style="background:{color};color:white;padding:1px 4px;'
        f'border-radius:3px" title="{cls}: {indices[0]}-{indices[-1]}">'
        f"{text}</span>"
    )
