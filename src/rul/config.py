"""Immutable experimental contract, fixed before opening official test labels."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class Config:
    seed: int = 42
    rul_cap: int = 125
    calibration_engines: int = 20
    cv_folds: int = 5
    windows: tuple = (5, 15, 30)
    alpha: float = 0.10
    n_jobs: int = 2
    bootstrap_repetitions: int = 2000
    min_history: int = 30

    @classmethod
    def load(cls, path="config.json"):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "windows" in values:
            values["windows"] = tuple(values["windows"])
        return cls(**values)

    def to_dict(self):
        return asdict(self)
