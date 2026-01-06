# core/config.py
"""
Configuration loading utilities.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional

from .schemas import AppConfig


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        AppConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if config_path is None:
        config_path = "configs/app.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return AppConfig(**config_dict)


def save_config(config: AppConfig, config_path: str) -> None:
    """
    Save application configuration to YAML file.
    
    Args:
        config: AppConfig object
        config_path: Path to save config file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
