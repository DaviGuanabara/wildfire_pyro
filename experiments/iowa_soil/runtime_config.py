from pathlib import Path
from config import load_full_config

CONFIG_PATH = Path(__file__).parent / "default_config.yaml"

_loaded = load_full_config(str(CONFIG_PATH))

BASE_RUN_PARAMETERS = _loaded.base_run_parameters
OPTUNA_CONFIG = _loaded.optuna
