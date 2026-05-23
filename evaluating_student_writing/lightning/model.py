import logging

import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoModelForTokenClassification

from evaluating_student_writing.lightning.constants import ID2LABEL, NUM_LABELS

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps or []
        self.lr_decay_factor = lr_decay_factor
        self.max_grad_norm = max_grad_norm

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
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("val_loss", outputs.loss, prog_bar=True)
        return {"val_loss": outputs.loss}

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
