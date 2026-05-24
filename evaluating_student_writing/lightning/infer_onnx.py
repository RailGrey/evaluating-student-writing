import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from evaluating_student_writing.lightning.constants import ID2LABEL
from evaluating_student_writing.lightning.data import EssayDataset, load_essays
from evaluating_student_writing.lightning.report import generate_report
from evaluating_student_writing.lightning.utils import (
    _get_git_commit_id,
    predictions_to_spans,
)
from metrics import evaluate

logger = logging.getLogger(__name__)


def predict_onnx(cfg: DictConfig) -> pd.DataFrame:
    import onnxruntime as ort

    model_dir = Path(cfg.paths.model_dir)
    onnx_path = model_dir / cfg.export.onnx_filename

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Run export.py first."
        )

    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.experiment_name)

    with mlflow.start_run(run_name="onnx_inference"):
        mlflow.log_param("git_commit_id", _get_git_commit_id())
        mlflow.log_param("onnx_model", str(onnx_path))
        mlflow.log_param("min_span_length", cfg.training.min_span_length)
        mlflow.log_param("max_length", cfg.features.max_length)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info("ONNX session created with providers: %s", session.get_providers())

        tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)

        test_dir = Path(cfg.paths.test_dir)
        texts = load_essays(test_dir)
        logger.info("Found %d test essays", len(texts))
        mlflow.log_param("test_essays", len(texts))

        dataset = EssayDataset(
            texts=texts,
            annotations=None,
            tokenizer=tokenizer,
            max_length=cfg.features.max_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_fn,
        )

        min_span_length = cfg.training.min_span_length
        all_rows = []

        for batch in loader:
            input_ids_np = batch["input_ids"].numpy()
            attention_mask_np = batch["attention_mask"].numpy()

            logits = session.run(
                None,
                {
                    "input_ids": input_ids_np,
                    "attention_mask": attention_mask_np,
                },
            )[0]
            preds = np.argmax(logits, axis=-1)

            for i in range(len(batch["essay_id"])):
                essay_id = batch["essay_id"][i]
                word_ids_list = batch["word_ids"][i]
                n_words = batch["n_words"][i]
                token_preds = preds[i].tolist()

                word_preds = []
                previous_word_idx = None
                for idx, word_idx in enumerate(word_ids_list):
                    if word_idx is None:
                        continue
                    if word_idx != previous_word_idx:
                        word_preds.append(ID2LABEL[token_preds[idx]])
                        previous_word_idx = word_idx

                word_preds = word_preds[:n_words]
                spans = predictions_to_spans(word_preds, min_span_length)

                for span in spans:
                    all_rows.append(
                        {
                            "id": essay_id,
                            "class": span["class"],
                            "predictionstring": span["predictionstring"],
                        }
                    )

        submission = pd.DataFrame(all_rows, columns=["id", "class", "predictionstring"])
        output_path = Path(cfg.paths.submission_path)
        results_dir = Path(cfg.paths.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        submission.to_csv(output_path, index=False)
        submission.to_csv(results_dir / "submission_onnx.csv", index=False)
        logger.info("Submission saved to %s: %d rows", output_path, len(submission))

        mlflow.log_metric("submission_rows", len(submission))
        mlflow.log_artifact(str(output_path))

        report_path = generate_report(
            texts=texts,
            submission=submission,
            output_dir=results_dir,
            filename="report_onnx.md",
        )
        mlflow.log_artifact(str(report_path))

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

            flat_metrics = {}
            for key, value in metrics_result.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict) and "f1" in sub_value:
                            flat_metrics[f"{key}/{sub_key}/f1"] = sub_value["f1"]
            mlflow.log_metrics(flat_metrics)

            generate_report(
                texts=texts,
                submission=submission,
                output_dir=results_dir,
                metrics_result=metrics_result,
                filename="report_onnx_with_metrics.md",
            )

    return submission


def _collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k in ("essay_id", "word_ids", "n_words"):
            result[k] = [item[k] for item in batch]
        else:
            import torch

            result[k] = torch.stack([item[k] for item in batch])
    return result
