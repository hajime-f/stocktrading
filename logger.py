import re
from logging import Formatter, LogRecord, config, getLogger

import yaml


class StripRichFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        formatted_log = super().format(record)
        stripped_log = re.sub(r"\[(/?[\w\s#]*)\]", "", formatted_log)

        return stripped_log


class Logger:
    def __init__(self, path="log_conf.yaml"):
        with open(path, "rt") as f:
            log_config = yaml.safe_load(f.read())
            config.dictConfig(log_config)
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
