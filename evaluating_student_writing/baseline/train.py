import json
from pathlib import Path

import joblib
import pandas as pd
from omegaconf import DictConfig
from xgboost import XGBClassifier

from evaluating_student_writing.baseline.data import (
    build_sentence_dataset,
    build_tfidf,
    split_dataset,
)
from evaluating_student_writing.baseline.utils import CLASSES, merge_segments
from metrics import evaluate


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


def train(cfg: DictConfig) -> None:
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(cfg.paths.train_csv)
    essays_dir = Path(cfg.paths.train_essays_dir)

    print("Building sentence dataset...")
    df = build_sentence_dataset(
        csv_path, essays_dir, overlap_threshold=cfg.preprocessing.overlap_threshold
    )
    print(f"Total sentences: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    train_df, val_df = split_dataset(
        df,
        test_size=cfg.preprocessing.val_size,
        random_state=cfg.seed,
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    print("Building TF-IDF features...")
    ngram_range = tuple(cfg.features.ngram_range)
    vectorizer, X_train, X_val = build_tfidf(
        train_df["sentence_text"],
        val_df["sentence_text"],
        max_features=cfg.features.max_features,
        ngram_range=ngram_range,
        sublinear_tf=cfg.features.sublinear_tf,
    )

    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_train = train_df["label"].map(label_to_idx)
    y_val = val_df["label"].map(label_to_idx)

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree,
        objective=cfg.model.objective,
        eval_metric=cfg.model.eval_metric,
        num_class=len(CLASSES),
        tree_method=cfg.model.tree_method,
        random_state=cfg.seed,
        n_jobs=cfg.model.n_jobs,
        verbose=cfg.model.verbose,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
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

    joblib.dump(model, model_dir / "xgb_model.joblib")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(label_to_idx, model_dir / "label_to_idx.joblib")
    print(f"\nModel saved to {model_dir}")


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def _entry(cfg: DictConfig) -> None:
        train(cfg)

    _entry()
