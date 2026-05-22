from collections import defaultdict

import pandas as pd

from metrics.base import _example_data, _match_group, _validate_cols, _word_set


def span_jaccard_iou(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    _validate_cols(gt_df, "gt_df")
    _validate_cols(pred_df, "pred_df")

    gt_grouped = gt_df.groupby(["id", "class"])["predictionstring"].apply(list)
    pred_grouped = pred_df.groupby(["id", "class"])["predictionstring"].apply(list)
    all_keys = set(gt_grouped.index) | set(pred_grouped.index)

    class_ious: dict[str, list[float]] = defaultdict(list)
    all_ious: list[float] = []

    for key in all_keys:
        cls = key[1]
        gt_sets = [_word_set(s) for s in gt_grouped.get(key, [])]
        pred_sets = [_word_set(s) for s in pred_grouped.get(key, [])]

        matched_pairs, _, _, _ = _match_group(gt_sets, pred_sets)

        for gi, pi in matched_pairs:
            gt = gt_sets[gi]
            pred = pred_sets[pi]
            union = len(gt | pred)
            iou = len(gt & pred) / union if union > 0 else 0.0
            class_ious[cls].append(iou)
            all_ious.append(iou)

    per_class = {}
    for cls in sorted(class_ious.keys()):
        ious = class_ious[cls]
        per_class[cls] = {
            "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
            "n_matched": len(ious),
        }

    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    return {"mean_iou": round(mean_iou, 4), "per_class": per_class}


if __name__ == "__main__":
    import json

    gt_df, pred_df = _example_data()
    result = span_jaccard_iou(gt_df, pred_df)
    print("Span Jaccard IoU (overlap quality of matched pairs):")
    print(json.dumps(result, indent=2))
