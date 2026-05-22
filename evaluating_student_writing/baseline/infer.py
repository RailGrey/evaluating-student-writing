from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

from evaluating_student_writing.baseline.utils import (
    CLASSES,
    ensure_nltk_data,
    get_sentence_word_ranges,
    merge_segments,
    split_sentences,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"


def predict(
    test_dir: Path,
    model_path: Path | None = None,
    vectorizer_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    if model_path is None:
        model_path = MODEL_DIR / "xgb_model.joblib"
    if vectorizer_path is None:
        vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "submission.csv"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

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
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}: {len(submission)} rows")
    return submission


if __name__ == "__main__":
    test_dir = PROJECT_ROOT / "data" / "test"
    predict(test_dir)
