import yaml
import os
from dotenv import load_dotenv


class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        load_dotenv()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self._replace_env_placeholders(self.config)

    def _replace_env_placeholders(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._replace_env_placeholders(value)
            elif (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                config_dict[key] = os.getenv(env_var)

    def get(self, key_path):
        keys = key_path.split(".")
        value = self.config
        for key in keys:
            value = value[key]
        return value


cm = ConfigManager()
