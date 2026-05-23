import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info(OmegaConf.to_yaml(cfg))

    from evaluating_student_writing.baseline.train import train

    train(cfg)


if __name__ == "__main__":
    main()
