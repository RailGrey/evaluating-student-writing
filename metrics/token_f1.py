from collections import defaultdict

import pandas as pd

from metrics.base import (
    _build_per_class,
    _example_data,
    _validate_cols,
    _word_set,
    aggregate_averages,
)


def token_f1(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    _validate_cols(gt_df, "gt_df")
    _validate_cols(pred_df, "pred_df")

    gt_grouped = gt_df.groupby("id")[["class", "predictionstring"]].apply(
        lambda g: list(zip(g["class"], g["predictionstring"]))
    )
    pred_grouped = pred_df.groupby("id")[["class", "predictionstring"]].apply(
        lambda g: list(zip(g["class"], g["predictionstring"]))
    )
    all_ids = set(gt_grouped.index) | set(pred_grouped.index)

    class_tp: dict[str, int] = defaultdict(int)
    class_fp: dict[str, int] = defaultdict(int)
    class_fn: dict[str, int] = defaultdict(int)

    for eid in all_ids:
        gt_spans = gt_grouped.get(eid, [])
        pred_spans = pred_grouped.get(eid, [])

        gt_tokens: dict[int, str] = {}
        for cls, ps in gt_spans:
            for idx in _word_set(ps):
                gt_tokens[idx] = cls

        pred_tokens: dict[int, str] = {}
        for cls, ps in pred_spans:
            for idx in _word_set(ps):
                pred_tokens[idx] = cls

        all_tokens = set(gt_tokens.keys()) | set(pred_tokens.keys())
        for token_idx in all_tokens:
            gt_label = gt_tokens.get(token_idx, "O")
            pred_label = pred_tokens.get(token_idx, "O")

            if gt_label != "O" and pred_label == gt_label:
                class_tp[gt_label] += 1
            elif gt_label != "O" and pred_label != gt_label:
                class_fn[gt_label] += 1
                if pred_label != "O":
                    class_fp[pred_label] += 1
            elif gt_label == "O" and pred_label != "O":
                class_fp[pred_label] += 1

    per_class = _build_per_class(class_tp, class_fp, class_fn)
    averages = aggregate_averages(per_class)

    return {"per_class": per_class, "averages": averages}


if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    gt_df, pred_df = _example_data()
    result = token_f1(gt_df, pred_df)
    logger.info(
        "Token-level F1 (word-level classification):\n%s", json.dumps(result, indent=2)
    )
