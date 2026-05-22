import pandas as pd

from metrics.base import aggregate_averages
from metrics.competition_f1 import competition_f1
from metrics.span_exact_match import span_exact_match_f1
from metrics.span_jaccard import span_jaccard_iou
from metrics.token_f1 import token_f1


def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    return {
        "competition_f1": competition_f1(gt_df, pred_df),
        "token_f1": token_f1(gt_df, pred_df),
        "span_exact_match_f1": span_exact_match_f1(gt_df, pred_df),
        "span_jaccard_iou": span_jaccard_iou(gt_df, pred_df),
    }


__all__ = [
    "evaluate",
    "competition_f1",
    "token_f1",
    "span_exact_match_f1",
    "span_jaccard_iou",
    "aggregate_averages",
]
