"""Configuration management with YAML inheritance."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a single YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge two config dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, config_dir: Optional[str] = None) -> DictConfig:
    """Load config with inheritance support."""
    if config_dir is None:
        config_dir = os.path.dirname(config_path)
    
    config = load_yaml(config_path)
    
    # Handle defaults/inheritance
    if 'defaults' in config:
        base_config = {}
        for default in config['defaults']:
            if isinstance(default, str):
                base_path = os.path.join(config_dir, f"{default}.yaml")
                base_config = merge_configs(base_config, load_yaml(base_path))
            elif isinstance(default, dict):
                for key, value in default.items():
                    base_path = os.path.join(config_dir, f"{value}.yaml")
                    base_config = merge_configs(base_config, load_yaml(base_path))
        del config['defaults']
        config = merge_configs(base_config, config)
    
    return OmegaConf.create(config)
