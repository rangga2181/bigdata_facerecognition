"""
src/utils/config_loader.py
YAML configuration loader dengan merge support.
"""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config(dict):
    """
    Dictionary dengan akses via dot notation.
    Contoh: cfg.training.lr
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def get_nested(self, *keys, default=None):
        """Akses nested key. Contoh: cfg.get_nested('training', 'lr')"""
        val = self
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val


def load_config(config_path: str, base_config: str = "configs/default.yaml") -> Config:
    """
    Load YAML config dengan support merge ke default config.

    Args:
        config_path: path ke config file yang ingin diload
        base_config: path ke default config sebagai base

    Returns:
        Config instance (dict dengan dot notation access)
    """
    # Load base config
    base = {}
    base_path = Path(base_config)
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    # Load target config
    with open(config_path, "r", encoding="utf-8") as f:
        override = yaml.safe_load(f) or {}

    # Merge (override menimpa base)
    merged = _deep_merge(base, override)
    return Config(merged)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Merge dua dict secara rekursif (override wins)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def save_config(config: Dict, save_path: str):
    """Menyimpan config dict ke YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(dict(config), f, default_flow_style=False, allow_unicode=True)
    print(f"💾 Config saved to {save_path}")
