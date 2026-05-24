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

    ensure_dvc_data([cfg.paths.model_dir])

    from evaluating_student_writing.lightning.export import export_onnx

    export_onnx(cfg)


if __name__ == "__main__":
    main()
