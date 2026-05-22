import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from evaluating_student_writing.baseline.data import (
    build_sentence_dataset,
    build_tfidf,
    split_dataset,
)
from evaluating_student_writing.baseline.utils import CLASSES, merge_segments
from metrics import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"


def _val_to_submission(
    val_df: pd.DataFrame, y_pred: pd.Series, idx_to_label: dict[int, str]
) -> pd.DataFrame:
    pred_labels = [idx_to_label[int(p)] for p in y_pred]
    val_df = val_df.copy()
    val_df["pred_label"] = pred_labels

    rows = []
    for eid, group in val_df.groupby("id", sort=False):
        group = group.sort_values("sentence_idx")
        labels = group["pred_label"].tolist()
        word_ranges = list(zip(group["word_range_start"], group["word_range_end"]))
        segments = merge_segments(labels, word_ranges)
        for seg in segments:
            rows.append(
                {
                    "id": eid,
                    "class": seg["class"],
                    "predictionstring": seg["predictionstring"],
                }
            )
    return pd.DataFrame(rows, columns=["id", "class", "predictionstring"])


def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PROJECT_ROOT / "data" / "train.csv"
    essays_dir = PROJECT_ROOT / "data" / "train"

    print("Building sentence dataset...")
    df = build_sentence_dataset(csv_path, essays_dir)
    print(f"Total sentences: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    train_df, val_df = split_dataset(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    print("Building TF-IDF features...")
    vectorizer, X_train, X_val = build_tfidf(
        train_df["sentence_text"], val_df["sentence_text"]
    )

    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_train = train_df["label"].map(label_to_idx)
    y_val = val_df["label"].map(label_to_idx)

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=10,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(CLASSES),
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    y_pred = model.predict(X_val)

    print("\nConverting validation predictions to submission format...")
    pred_df = _val_to_submission(val_df, y_pred, idx_to_label)

    gt_all = pd.read_csv(csv_path)
    val_ids = set(val_df["id"].unique())
    gt_df = (
        gt_all[gt_all["id"].isin(val_ids)][["id", "discourse_type", "predictionstring"]]
        .rename(columns={"discourse_type": "class"})
        .copy()
    )

    print("\nEvaluating metrics...")
    metrics_result = evaluate(gt_df, pred_df)
    print(json.dumps(metrics_result, indent=2))

    joblib.dump(model, MODEL_DIR / "xgb_model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(label_to_idx, MODEL_DIR / "label_to_idx.joblib")
    print(f"\nModel saved to {MODEL_DIR}")


if __name__ == "__main__":
    train()
