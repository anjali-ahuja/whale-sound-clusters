"""Configuration loading and validation."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, tries:
            1. configs/local.yaml
            2. configs/default.yaml
            
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try local config first, then default
        local_path = Path("configs/local.yaml")
        default_path = Path("configs/default.yaml")
        
        if local_path.exists():
            config_path = str(local_path)
        elif default_path.exists():
            config_path = str(default_path)
        else:
            raise FileNotFoundError(
                "No config file found. Please create configs/local.yaml or ensure configs/default.yaml exists."
            )
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables in paths
    config = _expand_env_vars(config)
    
    return config


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in config values."""
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


