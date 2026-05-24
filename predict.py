import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from evaluating_student_writing.dvc import ensure_dvc_data

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info(OmegaConf.to_yaml(cfg))

    ensure_dvc_data(
        [
            cfg.paths.model_dir,
            cfg.paths.test_dir,
        ]
    )

    if "model_name" in cfg.model:
        from evaluating_student_writing.lightning.infer import predict

        predict(cfg)
    elif "n_estimators" in cfg.model or "objective" in cfg.model:
        from evaluating_student_writing.baseline.infer import predict

        predict(cfg)
    else:
        raise ValueError(
            "Unknown model config. Use model=xgboost for baseline "
            "or model=bigbird for Lightning."
        )


if __name__ == "__main__":
    main()
