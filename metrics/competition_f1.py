from collections import defaultdict

import pandas as pd

from metrics.base import (
    _build_per_class,
    _example_data,
    _match_group,
    _validate_cols,
    _word_set,
    aggregate_averages,
)


def competition_f1(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    _validate_cols(gt_df, "gt_df")
    _validate_cols(pred_df, "pred_df")

    gt_grouped = gt_df.groupby(["id", "class"])["predictionstring"].apply(list)
    pred_grouped = pred_df.groupby(["id", "class"])["predictionstring"].apply(list)
    all_keys = set(gt_grouped.index) | set(pred_grouped.index)

    class_tp: dict[str, int] = defaultdict(int)
    class_fp: dict[str, int] = defaultdict(int)
    class_fn: dict[str, int] = defaultdict(int)

    for key in all_keys:
        cls = key[1]
        gt_sets = [_word_set(s) for s in gt_grouped.get(key, [])]
        pred_sets = [_word_set(s) for s in pred_grouped.get(key, [])]
        _, tp, fp, fn = _match_group(gt_sets, pred_sets)
        class_tp[cls] += tp
        class_fp[cls] += fp
        class_fn[cls] += fn

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
    result = competition_f1(gt_df, pred_df)
    logger.info(
        "Competition F1 (0.5 overlap threshold):\n%s", json.dumps(result, indent=2)
    )
