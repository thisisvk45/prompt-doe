"""Prompt assembly from component configurations."""
from __future__ import annotations

from src.components import (
    COMPONENT_KEYS,
    ComponentVariant,
    load_components,
    load_task_config,
)


def build_few_shot_block(task_config: dict) -> str:
    """Build few-shot examples string from task config."""
    examples = task_config.get("few_shot_examples", [])
    if not examples:
        return ""

    lines = ["Here are some examples:\n"]
    for i, ex in enumerate(examples, 1):
        q = ex["question"]
        a = ex["answer"]
        if "options" in ex:
            opts = "\n".join(ex["options"])
            q = f"{q}\n{opts}"
        lines.append(f"Example {i}:")
        lines.append(f"Q: {q}")
        lines.append(f"A: {a}\n")
    return "\n".join(lines)


def assemble_prompt(
    flags: dict[str, bool],
    task_config: dict,
    question: str,
    components: dict[str, ComponentVariant] | None = None,
) -> str:
    """Assemble a prompt from component flags.

    Args:
        flags: dict mapping component key -> True (present) or False (absent).
               Keys must match COMPONENT_KEYS.
        task_config: task configuration dict (from tasks.yaml).
        question: the actual question/problem text.
        components: pre-loaded components (optional, loads from config if None).

    Returns:
        Assembled prompt string.
    """
    if components is None:
        components = load_components()

    parts = []

    for key in COMPONENT_KEYS:
        comp = components[key]
        active = flags.get(key, True)

        if key == "few_shot":
            if active:
                block = build_few_shot_block(task_config)
                if block:
                    parts.append(block)
            # When absent, we just skip few-shot examples (no placeholder needed)
            continue

        variant = comp.present if active else comp.absent
        parts.append(variant)

    # Append the actual question
    parts.append(f"Question: {question}")

    return "\n\n".join(parts)


def flags_from_vector(vector: list[int]) -> dict[str, bool]:
    """Convert a design matrix row (list of -1/+1 or 0/1) to component flags.

    -1 or 0 -> absent (False), +1 or 1 -> present (True).
    """
    flags = {}
    for key, val in zip(COMPONENT_KEYS, vector):
        flags[key] = val > 0
    return flags
