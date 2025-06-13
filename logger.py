import logging.config
import yaml


if __name__ == "__main__":
    with open("log_conf.yaml", "rt") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
