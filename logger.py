from logging import getLogger
import logging.config
import yaml


class Logger:
    def __init__(self, path="log_conf.yaml"):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        self.logger = getLogger(__name__)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)


if __name__ == "__main__":
    with open("log_conf.yaml", "rt") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

    logger = getLogger(__name__)
    logger.info("これはテストです。")
