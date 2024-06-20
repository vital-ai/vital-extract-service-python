import logging
import yaml


class ConfigUtils:

    @staticmethod
    def load_config():
        with open("../extract_service_config.yaml", "r") as config_stream:
            try:
                return yaml.safe_load(config_stream)
            except yaml.YAMLError as exc:
                logger = logging.getLogger("ExtractServiceLogger")
                logger.info("failed to load config file")
