import json
import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluating_student_writing.lightning.constants import ID2LABEL
from evaluating_student_writing.lightning.data import (
    EssayDataset,
    load_annotations,
    load_essays,
    split_essay_ids,
)
from evaluating_student_writing.lightning.model import NERLightningModule
from evaluating_student_writing.lightning.utils import predictions_to_spans
from metrics import evaluate

logger = logging.getLogger(__name__)


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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    model = NERLightningModule(
        model_name=cfg.model.model_name,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        learning_rate=cfg.training.learning_rate,
        lr_decay_steps=list(cfg.training.lr_decay_steps),
        lr_decay_factor=cfg.training.lr_decay_factor,
        max_grad_norm=cfg.training.max_grad_norm,
        cache_dir=hf_cache,
    )

    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        gradient_clip_val=cfg.training.max_grad_norm,
        val_check_interval=cfg.training.val_interval,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_loader, val_loader)

    logger.info("Running validation predictions for metrics...")
    pred_df = _predict_val(model, tokenizer, val_texts, cfg)

    gt_all = pd.read_csv(Path(cfg.paths.train_csv))
    gt_df = (
        gt_all[gt_all["id"].isin(val_ids)][["id", "discourse_type", "predictionstring"]]
        .rename(columns={"discourse_type": "class"})
        .copy()
    )

    logger.info("Evaluating metrics...")
    metrics_result = evaluate(gt_df, pred_df)
    logger.info("Metrics result:\n%s", json.dumps(metrics_result, indent=2))

    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(model_dir / "bigbird")
    tokenizer.save_pretrained(model_dir / "bigbird")
    logger.info("Model saved to %s/bigbird", model_dir)


def _predict_val(
    model: NERLightningModule,
    tokenizer: AutoTokenizer,
    texts: dict[str, str],
    cfg: DictConfig,
) -> pd.DataFrame:
    model.eval()
    device = next(model.parameters()).device

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

    all_rows = []
    min_span_length = cfg.training.min_span_length

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
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

    return pd.DataFrame(all_rows, columns=["id", "class", "predictionstring"])


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
