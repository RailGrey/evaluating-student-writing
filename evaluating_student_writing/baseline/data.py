from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from evaluating_student_writing.baseline.utils import (
    CLASSES,
    assign_sentence_labels,
    ensure_nltk_data,
    get_sentence_word_ranges,
    split_sentences,
)


def load_essay(essay_id: str, essays_dir: Path) -> str:
    path = essays_dir / f"{essay_id}.txt"
    return path.read_text(encoding="utf-8")


def build_sentence_dataset(
    csv_path: Path,
    essays_dir: Path,
    overlap_threshold: float = 0.3,
) -> pd.DataFrame:
    ensure_nltk_data()
    df = pd.read_csv(csv_path)
    essays_by_id = {}
    annotations_by_id: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        eid = row["id"]
        if eid not in annotations_by_id:
            annotations_by_id[eid] = []
        annotations_by_id[eid].append(
            {
                "discourse_start": int(row["discourse_start"]),
                "discourse_end": int(row["discourse_end"]),
                "discourse_type": row["discourse_type"],
            }
        )

    records = []
    for eid in tqdm(annotations_by_id, desc="Processing essays"):
        essay_text = load_essay(eid, essays_dir)
        sentences = split_sentences(essay_text)
        labels = assign_sentence_labels(
            essay_text, sentences, annotations_by_id[eid], overlap_threshold
        )
        word_ranges = get_sentence_word_ranges(essay_text, sentences)
        for sent_idx, (sent_text, label, wr) in enumerate(
            zip(sentences, labels, word_ranges)
        ):
            records.append(
                {
                    "id": eid,
                    "sentence_idx": sent_idx,
                    "sentence_text": sent_text,
                    "label": label,
                    "word_range_start": wr[0],
                    "word_range_end": wr[1],
                }
            )
        essays_by_id[eid] = essay_text

    return pd.DataFrame(records)


def split_dataset(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = next(splitter.split(df, groups=df["id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(
        drop=True
    )


def build_tfidf(
    train_texts: pd.Series,
    val_texts: pd.Series,
    max_features: int,
    ngram_range: tuple[int, int],
    sublinear_tf: bool,
) -> tuple[TfidfVectorizer, pd.DataFrame, pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    return vectorizer, X_train, X_val
