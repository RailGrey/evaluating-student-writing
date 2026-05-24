import json
import logging
from pathlib import Path

import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from evaluating_student_writing.lightning.constants import ID2LABEL
from evaluating_student_writing.lightning.data import EssayDataset, load_essays
from evaluating_student_writing.lightning.report import generate_report
from evaluating_student_writing.lightning.utils import (
    _get_git_commit_id,
    predictions_to_spans,
)
from metrics import evaluate

logger = logging.getLogger(__name__)


def predict(cfg: DictConfig) -> pd.DataFrame:
    if "model_name" not in cfg.model:
        raise ValueError(
            "Lightning infer requires a NER model config. "
            "Run with: model=bigbird features=tokenizer"
        )

    from evaluating_student_writing.dvc import ensure_dvc_data

    ensure_dvc_data([cfg.paths.model_dir, cfg.paths.test_dir])

    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.experiment_name)

    with mlflow.start_run(run_name="inference"):
        mlflow.log_param("git_commit_id", _get_git_commit_id())
        mlflow.log_param("model_dir", str(cfg.paths.model_dir))
        mlflow.log_param("test_dir", str(cfg.paths.test_dir))
        mlflow.log_param("min_span_length", cfg.training.min_span_length)
        mlflow.log_param("max_length", cfg.features.max_length)

        model_dir = Path(cfg.paths.model_dir) / "bigbird"
        output_path = Path(cfg.paths.submission_path)
        test_dir = Path(cfg.paths.test_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)

        config = AutoConfig.from_pretrained(model_dir)
        hf_model = AutoModelForTokenClassification.from_pretrained(
            model_dir, config=config
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_model.to(device)
        hf_model.eval()

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

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                logits = hf_model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                preds = torch.argmax(logits, dim=-1)

                for i in range(len(batch["essay_id"])):
                    essay_id = batch["essay_id"][i]
                    word_ids_list = batch["word_ids"][i]
                    n_words = batch["n_words"][i]
                    token_preds = preds[i].cpu().tolist()

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
        submission.to_csv(results_dir / "submission.csv", index=False)
        logger.info("Submission saved to %s: %d rows", output_path, len(submission))
        logger.info("Submission also copied to %s/submission.csv", results_dir)

        mlflow.log_metric("submission_rows", len(submission))
        mlflow.log_artifact(str(output_path))

        report_path = generate_report(
            texts=texts,
            submission=submission,
            output_dir=Path(cfg.paths.results_dir),
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

            metrics_report_path = generate_report(
                texts=texts,
                submission=submission,
                output_dir=Path(cfg.paths.results_dir),
                metrics_result=metrics_result,
            )
            mlflow.log_artifact(str(metrics_report_path))

    return submission


def _collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k in ("essay_id", "word_ids", "n_words"):
            result[k] = [item[k] for item in batch]
        else:
            result[k] = torch.stack([item[k] for item in batch])
    return result


if __name__ == "__main__":
    import logging
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    import hydra

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def _entry(cfg: DictConfig) -> None:
        predict(cfg)

    _entry()
