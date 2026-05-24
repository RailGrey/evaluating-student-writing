import logging
from pathlib import Path

import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger as PLMLFlowLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from evaluating_student_writing.lightning.data import (
    EssayDataset,
    load_annotations,
    load_essays,
    split_essay_ids,
)
from evaluating_student_writing.lightning.model import NERLightningModule
from evaluating_student_writing.lightning.plot_utils import _generate_plots
from evaluating_student_writing.lightning.utils import (
    _collate_fn,
    _flatten_metrics,
    _log_experiment_params,
    predictions_to_spans,
)
from metrics import evaluate

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    if "model_name" not in cfg.model:
        raise ValueError(
            "Lightning train requires a NER model config. "
            "Run with: model=bigbird features=tokenizer"
        )

    hf_cache = str(Path(cfg.paths.hf_cache_dir).resolve())

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name, add_prefix_space=True, cache_dir=hf_cache
    )

    texts = load_essays(Path(cfg.paths.train_essays_dir))
    annotations = load_annotations(Path(cfg.paths.train_csv))

    all_ids = sorted(texts.keys())
    train_ids, val_ids = split_essay_ids(
        all_ids, val_size=cfg.training.val_size, seed=cfg.seed
    )
    logger.info("Train essays: %d, Val essays: %d", len(train_ids), len(val_ids))

    train_texts = {eid: texts[eid] for eid in train_ids}
    val_texts = {eid: texts[eid] for eid in val_ids}

    gt_all = pd.read_csv(Path(cfg.paths.train_csv))
    val_gt_df = (
        gt_all[gt_all["id"].isin(val_ids)][["id", "discourse_type", "predictionstring"]]
        .rename(columns={"discourse_type": "class"})
        .copy()
    )

    train_dataset = EssayDataset(
        texts=train_texts,
        annotations=annotations,
        tokenizer=tokenizer,
        max_length=cfg.features.max_length,
        label_all_subtokens=cfg.features.label_all_subtokens,
    )
    val_dataset = EssayDataset(
        texts=val_texts,
        annotations=annotations,
        tokenizer=tokenizer,
        max_length=cfg.features.max_length,
        label_all_subtokens=cfg.features.label_all_subtokens,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        persistent_workers=True,
    )

    model = NERLightningModule(
        model_name=cfg.model.model_name,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        learning_rate=cfg.training.learning_rate,
        lr_decay_steps=list(cfg.training.lr_decay_steps),
        lr_decay_factor=cfg.training.lr_decay_factor,
        max_grad_norm=cfg.training.max_grad_norm,
        cache_dir=hf_cache,
        val_gt_df=val_gt_df,
        min_span_length=cfg.training.min_span_length,
    )

    pl_logger = PLMLFlowLogger(
        experiment_name=cfg.experiment.experiment_name,
        tracking_uri=cfg.experiment.tracking_uri,
        run_name=cfg.experiment.run_name,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=cfg.training.early_stopping_callback_delta,
        patience=cfg.training.early_stopping_callback_patience,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        gradient_clip_val=cfg.training.max_grad_norm,
        val_check_interval=cfg.training.val_interval,
        limit_val_batches=cfg.training.limit_val_batches,
        enable_checkpointing=True,
        logger=pl_logger,
        callbacks=[early_stop_callback],
    )

    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.experiment_name)

    with mlflow.start_run(run_id=pl_logger.run_id if pl_logger.run_id else None):
        _log_experiment_params(cfg)
        mlflow.log_param("train_essays", len(train_ids))
        mlflow.log_param("val_essays", len(val_ids))

        trainer.fit(model, train_loader, val_loader)

        plots_dir = Path(cfg.paths.plots_dir)
        client = MlflowClient(tracking_uri=cfg.experiment.tracking_uri)
        run_id = mlflow.active_run().info.run_id
        _generate_plots(client, run_id, None, plots_dir)

        model_dir = Path(cfg.paths.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "bigbird"
        model.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    import logging

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    import hydra

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def _entry(cfg: DictConfig) -> None:
        train(cfg)

    _entry()
