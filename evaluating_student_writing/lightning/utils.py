import logging
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def predictions_to_spans(word_preds: list[str], min_span_length: int) -> list[dict]:
    spans = []
    j = 0
    while j < len(word_preds):
        cls = word_preds[j]
        if cls == "O":
            j += 1
            continue

        cls_normalized = cls.replace("B-", "I-")
        end = j + 1
        while end < len(word_preds) and word_preds[end] == cls_normalized:
            end += 1

        span_length = end - j
        if span_length >= min_span_length:
            class_name = cls_normalized.replace("I-", "")
            word_indices = list(range(j, end))
            spans.append(
                {
                    "class": class_name,
                    "predictionstring": " ".join(map(str, word_indices)),
                }
            )
        j = end

    return spans


def _collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k in ("essay_id", "word_ids", "n_words"):
            result[k] = [item[k] for item in batch]
        else:
            result[k] = torch.stack([item[k] for item in batch])
    return result


def _collate_infer(batch):
    return _collate_fn(batch)


def _get_git_commit_id() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _flatten_metrics(metrics_result: dict, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics_result.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, f"{full_key}/"))
        elif isinstance(value, (int, float)):
            flat[full_key] = value
    return flat


def _log_experiment_params(cfg: DictConfig) -> None:
    params = OmegaConf.to_container(cfg, resolve=True)
    flat_params: dict[str, str] = {}
    _flatten_dict(params, "", flat_params)
    mlflow.log_params(flat_params)

    commit_id = _get_git_commit_id()
    mlflow.log_param("git_commit_id", commit_id)
    logger.info("Git commit id: %s", commit_id)


def _flatten_dict(d: dict | list, prefix: str, out: dict[str, str]) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            _flatten_dict(v, f"{prefix}{k}/", out)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            _flatten_dict(v, f"{prefix}{i}/", out)
    else:
        out[prefix.rstrip("/")] = str(d)
