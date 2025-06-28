import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return its contents."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.error(f"Configuration file not found: {cfg_path}")
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    try:
        with cfg_path.open("r") as f:
            config = yaml.safe_load(f)
        if config is None:
            logger.error(f"Configuration file is empty: {cfg_path}")
            raise ValueError(f"Configuration file is empty: {cfg_path}")
        logger.info(f"Loaded configuration from {cfg_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML format in {cfg_path}: {e}")
        raise
