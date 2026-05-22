from collections import defaultdict

import pandas as pd

from metrics.base import (
    _build_per_class,
    _example_data,
    _validate_cols,
    _word_set,
    aggregate_averages,
)


def span_exact_match_f1(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
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

        used_gt: set[int] = set()
        used_pred: set[int] = set()

        for pi, pred in enumerate(pred_sets):
            if pi in used_pred:
                continue
            for gi, gt in enumerate(gt_sets):
                if gi in used_gt:
                    continue
                if gt == pred:
                    class_tp[cls] += 1
                    used_gt.add(gi)
                    used_pred.add(pi)
                    break
            else:
                class_fp[cls] += 1

        class_fn[cls] += len(gt_sets) - len(used_gt)

    per_class = _build_per_class(class_tp, class_fp, class_fn)
    averages = aggregate_averages(per_class)

    return {"per_class": per_class, "averages": averages}


if __name__ == "__main__":
    import json

    gt_df, pred_df = _example_data()
    result = span_exact_match_f1(gt_df, pred_df)
    print("Span Exact Match F1 (exact word set equality):")
    print(json.dumps(result, indent=2))
