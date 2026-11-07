"""Immutable experimental contract, fixed before opening official test labels."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math


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

    def __post_init__(self):
        """Reject unsupported experiments before creating any output files.

        Public artifacts explicitly describe nominal 90% intervals. Changing
        coverage requires a new schema and presentation, not just a parameter.
        """
        for name in ("seed", "rul_cap", "calibration_engines", "cv_folds",
                     "bootstrap_repetitions", "min_history", "n_jobs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} debe ser entero.")
        if self.seed < 0 or min(self.rul_cap, self.bootstrap_repetitions, self.min_history) < 1:
            raise ValueError("Semilla no negativa y límites/repeticiones positivos requeridos.")
        if self.cv_folds < 2 or self.n_jobs == 0 or self.calibration_engines < 9:
            raise ValueError("Se requieren al menos 2 folds, 9 motores de calibración y n_jobs distinto de 0.")
        if not math.isfinite(self.alpha) or self.alpha != 0.10:
            raise ValueError("Este estudio y su esquema lower90/upper90 requieren alpha=0.10.")
        if not self.windows or len(set(self.windows)) != len(self.windows) or any(
            not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in self.windows
        ):
            raise ValueError("Las ventanas deben ser enteros positivos sin repeticiones.")

    @classmethod
    def load(cls, path="config.json"):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "windows" in values:
            values["windows"] = tuple(values["windows"])
        return cls(**values)

    def to_dict(self):
        return asdict(self)
