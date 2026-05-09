from __future__ import annotations
"""Prompt component definitions and management."""

import yaml
from pathlib import Path
from dataclasses import dataclass

CONFIG_DIR = Path(__file__).parent.parent / "config"

COMPONENT_KEYS = [
    "system_role",
    "persona",
    "few_shot",
    "cot_trigger",
    "output_format",
    "constraints",
]


@dataclass
class ComponentVariant:
    """A prompt component with present/absent variants."""
    key: str
    name: str
    present: str
    absent: str


def load_components() -> dict[str, ComponentVariant]:
    """Load component definitions from config."""
    with open(CONFIG_DIR / "components.yaml") as f:
        raw = yaml.safe_load(f)

    components = {}
    for key in COMPONENT_KEYS:
        cfg = raw[key]
        present = cfg["present"]
        absent = cfg["absent"]
        # few_shot is special: bool flag, actual examples come from task config
        if isinstance(present, bool):
            present = present
            absent = absent
        components[key] = ComponentVariant(
            key=key,
            name=cfg["name"],
            present=str(present).strip() if not isinstance(present, bool) else present,
            absent=str(absent).strip() if not isinstance(absent, bool) else absent,
        )
    return components


def load_task_config(task_name: str) -> dict:
    """Load task configuration."""
    with open(CONFIG_DIR / "tasks.yaml") as f:
        tasks = yaml.safe_load(f)
    return tasks[task_name]


def load_model_config(model_name: str) -> dict:
    """Load model configuration."""
    with open(CONFIG_DIR / "models.yaml") as f:
        models = yaml.safe_load(f)
    return models[model_name]
