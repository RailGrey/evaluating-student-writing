import json
from pathlib import Path

import joblib
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from evaluating_student_writing.baseline.utils import (
    CLASSES,
    ensure_nltk_data,
    get_sentence_word_ranges,
    merge_segments,
    split_sentences,
)
from metrics import evaluate


def predict(cfg: DictConfig) -> pd.DataFrame:
    model_dir = Path(cfg.paths.model_dir)
    output_path = Path(cfg.paths.submission_path)
    test_dir = Path(cfg.paths.test_dir)

    model = joblib.load(model_dir / "xgb_model.joblib")
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")

    ensure_nltk_data()
    test_files = sorted(test_dir.glob("*.txt"))
    print(f"Found {len(test_files)} test essays")

    rows = []
    for tf in tqdm(test_files, desc="Predicting essays"):
        essay_id = tf.stem
        essay_text = tf.read_text(encoding="utf-8")
        sentences = split_sentences(essay_text)

        if not sentences:
            continue

        X = vectorizer.transform(sentences)
        preds = model.predict(X)
        pred_labels = [CLASSES[int(p)] for p in preds]

        word_ranges = get_sentence_word_ranges(essay_text, sentences)
        segments = merge_segments(pred_labels, word_ranges)

        for seg in segments:
            rows.append(
                {
                    "id": essay_id,
                    "class": seg["class"],
                    "predictionstring": seg["predictionstring"],
                }
            )

    submission = pd.DataFrame(rows, columns=["id", "class", "predictionstring"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}: {len(submission)} rows")

    if cfg.paths.get("gt_csv_path"):
        gt_all = pd.read_csv(cfg.paths.gt_csv_path)
        pred_ids = set(submission["id"].unique())
        gt_df = (
            gt_all[gt_all["id"].isin(pred_ids)][
                ["id", "discourse_type", "predictionstring"]
            ]
            .rename(columns={"discourse_type": "class"})
            .copy()
        )
        print("\nEvaluating metrics...")
        metrics_result = evaluate(gt_df, submission)
        print(json.dumps(metrics_result, indent=2))

    return submission


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def _entry(cfg: DictConfig) -> None:
        predict(cfg)

    _entry()
