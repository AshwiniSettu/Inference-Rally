"""Load ``racers.yaml`` and build the racers for the active profile.

Profiles let the same code run with simulated racers (UI work), local Mac models,
or Snapdragon NPU/GPU/CPU models — switching is a one-line config change.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from inference_rally.deciders import Decider, HumanDecider, OpenAIDecider, SimulatedDecider
from inference_rally.errors import ConfigError, DeciderError
from inference_rally.maze import Maze
from inference_rally.models import RacerSpec, RaceSettings

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "racers.yaml"


class AppConfig(BaseModel):
    """Parsed ``racers.yaml``."""

    race: RaceSettings
    active_profile: str
    profiles: dict[str, list[RacerSpec]]

    def racers(self, profile: str | None = None) -> list[RacerSpec]:
        """Racer specs for ``profile`` (default: the active one)."""
        name = profile or self.active_profile
        if name not in self.profiles:
            raise ConfigError(f"Unknown profile {name!r}; choose from {sorted(self.profiles)}")
        return self.profiles[name]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read and validate the YAML config.

    Raises:
        ConfigError: If the file is missing or does not match the schema.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return AppConfig.model_validate(raw)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"Invalid config {path}: {exc}") from exc


def build_decider(spec: RacerSpec, maze: Maze, seed: int) -> Decider:
    """Create the decider that will drive ``spec``'s car.

    Args:
        spec: Racer definition from the config.
        maze: Current track (simulated racers use it to judge moves).
        seed: Race seed; combined with the racer id so racers behave independently.

    Raises:
        ConfigError: If the racer's runtime is not available yet or misconfigured.
    """
    if spec.runtime == "simulated":
        if spec.sim is None:
            raise ConfigError(f"Racer {spec.id!r} is simulated but has no 'sim' profile")
        return SimulatedDecider(spec.sim, maze, random.Random(f"{seed}:{spec.id}"))
    if spec.runtime == "human":
        return HumanDecider()
    try:
        return OpenAIDecider(spec)
    except DeciderError as exc:
        raise ConfigError(str(exc)) from exc
