import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForTokenClassification

from evaluating_student_writing.lightning.constants import ID2LABEL, NUM_LABELS

logger = logging.getLogger(__name__)


def export_onnx(cfg: DictConfig) -> Path:
    if "model_name" not in cfg.model:
        raise ValueError("ONNX export requires a NER model config (model=bigbird).")

    model_dir = Path(cfg.paths.model_dir)
    onnx_path = model_dir / cfg.export.onnx_filename

    hf_cache = str(Path(cfg.paths.hf_cache_dir).resolve())
    max_length = cfg.features.max_length
    opset = cfg.export.get("opset_version", 17)

    config = AutoConfig.from_pretrained(cfg.model.model_name, cache_dir=hf_cache)
    config.num_labels = NUM_LABELS
    config.id2label = {i: ID2LABEL[i] for i in range(NUM_LABELS)}
    config.label2id = {v: k for k, v in config.id2label.items()}
    config.attention_type = "original_full"

    logger.info("Loading model with attention_type='original_full' from %s", model_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        config=config,
        ignore_mismatched_sizes=True,
        cache_dir=hf_cache,
    )
    model.eval()

    dummy_input_ids = torch.randint(0, config.vocab_size, (1, max_length))
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)

    logger.info("Exporting ONNX to %s (opset=%d)", onnx_path, opset)
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
    )
    logger.info("ONNX export complete: %s", onnx_path)
    return onnx_path
