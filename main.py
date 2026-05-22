import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    from evaluating_student_writing.baseline.train import train

    train(cfg)


if __name__ == "__main__":
    main()
