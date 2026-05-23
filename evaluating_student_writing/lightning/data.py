import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from evaluating_student_writing.lightning.constants import LABEL2ID

logger = logging.getLogger(__name__)


class EssayDataset(Dataset):
    def __init__(
        self,
        texts: dict[str, str],
        annotations: dict[str, list[dict]] | None,
        tokenizer: AutoTokenizer,
        max_length: int,
        label_all_subtokens: bool = True,
    ):
        self.ids = sorted(texts.keys())
        self.texts = texts
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_all_subtokens = label_all_subtokens

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        essay_id = self.ids[idx]
        text = self.texts[essay_id]
        words = text.split()

        word_labels = self._get_word_labels(essay_id, len(words))

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(0) if hasattr(encoding, "word_ids") else None

        if not self.tokenizer.is_fast:
            encoded_dict = self.tokenizer(
                words,
                is_split_into_words=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            word_ids_list = encoded_dict.word_ids()
        else:
            word_ids_list = encoding.word_ids(0)

        labels = []
        previous_word_idx = None
        for word_idx in word_ids_list:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(LABEL2ID[word_labels[word_idx]])
            else:
                if self.label_all_subtokens:
                    labels.append(LABEL2ID[word_labels[word_idx]])
                else:
                    labels.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "word_ids": word_ids_list,
            "n_words": len(words),
            "essay_id": essay_id,
        }

    def _get_word_labels(self, essay_id: str, n_words: int) -> list[str]:
        labels = ["O"] * n_words

        if self.annotations is None:
            return labels

        for ann in self.annotations.get(essay_id, []):
            word_indices = ann["word_indices"]
            discourse_type = ann["discourse_type"]

            if word_indices and 0 <= word_indices[0] < n_words:
                labels[word_indices[0]] = f"B-{discourse_type}"
            for k in word_indices[1:]:
                if 0 <= k < n_words:
                    labels[k] = f"I-{discourse_type}"

        return labels


def load_essays(essays_dir: Path) -> dict[str, str]:
    texts = {}
    for f in sorted(essays_dir.glob("*.txt")):
        texts[f.stem] = f.read_text(encoding="utf-8")
    return texts


def load_annotations(csv_path: Path) -> dict[str, list[dict]]:
    df = pd.read_csv(csv_path)
    annotations: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        eid = row["id"]
        if eid not in annotations:
            annotations[eid] = []

        ps = row["predictionstring"]
        word_indices = [int(x) for x in str(ps).split()] if pd.notna(ps) else []

        annotations[eid].append(
            {
                "discourse_type": row["discourse_type"],
                "word_indices": word_indices,
            }
        )

    return annotations


def split_essay_ids(
    essay_ids: list[str], val_size: float, seed: int
) -> tuple[list[str], list[str]]:
    import numpy as np

    rng = np.random.RandomState(seed)
    ids = list(essay_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_size)
    return sorted(ids[n_val:]), sorted(ids[:n_val])
