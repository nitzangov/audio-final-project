"""YAML configuration loader with nested attribute access."""

from pathlib import Path
import yaml


class Config:
    """Recursive namespace wrapping a nested dict for dot-access."""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self):
        return f"Config({self.to_dict()})"


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a Config namespace."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(raw)
