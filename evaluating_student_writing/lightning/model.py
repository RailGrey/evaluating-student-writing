import logging

import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoModelForTokenClassification

from evaluating_student_writing.lightning.constants import ID2LABEL, NUM_LABELS
from evaluating_student_writing.lightning.utils import predictions_to_spans
from metrics import evaluate

logger = logging.getLogger(__name__)


class NERLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
        hidden_dropout_prob: float = 0.1,
        learning_rate: float = 2.5e-5,
        lr_decay_steps: list[int] | None = None,
        lr_decay_factor: float = 0.1,
        max_grad_norm: float = 10.0,
        cache_dir: str | None = None,
        val_gt_df: pd.DataFrame | None = None,
        min_span_length: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps or []
        self.lr_decay_factor = lr_decay_factor
        self.max_grad_norm = max_grad_norm
        self.val_gt_df = val_gt_df
        self.min_span_length = min_span_length
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.val_pred_rows = []

        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        config.num_labels = num_labels
        config.hidden_dropout_prob = hidden_dropout_prob
        config.id2label = {i: ID2LABEL[i] for i in range(num_labels)}
        config.label2id = {v: k for k, v in config.id2label.items()}

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True, cache_dir=cache_dir
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.train_step_outputs.append(outputs.loss)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True, on_step=True)
        return outputs.loss

    def on_validation_start(self):
        if self.train_step_outputs:
            avg_loss = torch.stack(self.train_step_outputs).mean()
            self.log("train_loss", avg_loss, prog_bar=True)
        self.train_step_outputs.clear()
        self.val_pred_rows.clear()

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.validation_step_outputs.append(outputs.loss)

        preds = torch.argmax(outputs.logits, dim=-1)
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
            spans = predictions_to_spans(word_preds, self.min_span_length)

            for span in spans:
                self.val_pred_rows.append(
                    {
                        "id": essay_id,
                        "class": span["class"],
                        "predictionstring": span["predictionstring"],
                    }
                )

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

        if self.val_gt_df is not None and self.val_pred_rows:
            pred_df = pd.DataFrame(
                self.val_pred_rows, columns=["id", "class", "predictionstring"]
            )
            pred_ids = set(pred_df["id"].unique())
            gt_filtered = self.val_gt_df[self.val_gt_df["id"].isin(pred_ids)]

            metrics_result = evaluate(gt_filtered, pred_df)
            metric_names = [
                "competition_f1",
                "token_f1",
                "span_exact_match_f1",
                "span_jaccard_iou",
            ]
            for m in metric_names:
                macro_f1 = (
                    metrics_result.get(m, {})
                    .get("averages", {})
                    .get("macro", {})
                    .get("f1", 0.0)
                )
                self.log(f"{m}/macro_f1", macro_f1, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if not self.lr_decay_steps:
            return optimizer

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.lr_decay_steps,
            gamma=self.lr_decay_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return torch.argmax(outputs.logits, dim=-1)
