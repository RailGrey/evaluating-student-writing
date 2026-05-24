import json
import logging
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

logger = logging.getLogger(__name__)


def predict(cfg: DictConfig) -> pd.DataFrame:
    model_dir = Path(cfg.paths.model_dir)
    output_path = Path(cfg.paths.submission_path)
    test_dir = Path(cfg.paths.test_dir)

    model = joblib.load(model_dir / "xgb_model.joblib")
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")

    ensure_nltk_data()
    test_files = sorted(test_dir.glob("*.txt"))
    logger.info("Found %d test essays", len(test_files))

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
    output_path = Path(cfg.paths.submission_path)
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    submission.to_csv(results_dir / "baseline_submission.csv", index=False)
    logger.info("Submission saved to %s: %d rows", output_path, len(submission))

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
        logger.info("Evaluating metrics...")
        metrics_result = evaluate(gt_df, submission)
        logger.info("Metrics result:\n%s", json.dumps(metrics_result, indent=2))

        metrics_path = results_dir / "baseline_metrics.json"
        metrics_path.write_text(json.dumps(metrics_result, indent=2), encoding="utf-8")
        logger.info("Metrics saved to %s", metrics_path)

    return submission


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    import hydra

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def _entry(cfg: DictConfig) -> None:
        predict(cfg)

    _entry()
