# MIT License - Copyright (c) 2026 Vikas Kumar
from __future__ import annotations
"""Data loaders for GSM8K, BBH-DateUnderstanding, and MMLU-Pro."""

import re
import random
from datasets import load_dataset
from src.components import load_task_config


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract numeric answer from GSM8K answer string (after ####)."""
    match = re.search(r"####\s*([\d,\.\-]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.strip()


def format_mmlu_question(row: dict) -> str:
    """Format MMLU-Pro question with lettered options."""
    question = row["question"]
    options = row["options"]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_lines = [f"({letters[i]}) {opt}" for i, opt in enumerate(options)]
    return f"{question}\n" + "\n".join(option_lines)


def load_task_data(task_name: str) -> list[dict]:
    """Load and sample examples for a task.

    Returns list of dicts with keys: 'question', 'answer', 'raw'.
    """
    config = load_task_config(task_name)
    n = config["n_samples"]
    seed = config["seed"]

    ds_kwargs = {"path": config["dataset"], "split": config["split"]}
    if config.get("subset"):
        ds_kwargs["name"] = config["subset"]

    ds = load_dataset(**ds_kwargs)

    # Sample
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    sampled = [ds[i] for i in indices]

    examples = []
    for row in sampled:
        if task_name == "gsm8k":
            question = row[config["question_field"]]
            answer = extract_gsm8k_answer(row[config["answer_field"]])
        elif task_name == "bbh_date":
            question = row[config["question_field"]]
            answer = row[config["answer_field"]].strip()
        elif task_name == "mmlu_pro":
            question = format_mmlu_question(row)
            # MMLU-Pro answer is a bare letter like 'I' — wrap in parens
            raw_ans = row[config["answer_field"]].strip()
            answer = f"({raw_ans})" if not raw_ans.startswith("(") else raw_ans
        else:
            raise ValueError(f"Unknown task: {task_name}")

        examples.append({
            "question": question,
            "answer": answer,
            "raw": row,
        })

    return examples


def load_task_data_with_seed(task_name: str, seed: int) -> list[dict]:
    """Load task data with a custom seed (for different example subsets)."""
    config = load_task_config(task_name)
    n = config["n_samples"]

    ds_kwargs = {"path": config["dataset"], "split": config["split"]}
    if config.get("subset"):
        ds_kwargs["name"] = config["subset"]

    ds = load_dataset(**ds_kwargs)

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    sampled = [ds[i] for i in indices]

    examples = []
    for row in sampled:
        if task_name == "gsm8k":
            question = row[config["question_field"]]
            answer = extract_gsm8k_answer(row[config["answer_field"]])
        elif task_name == "bbh_date":
            question = row[config["question_field"]]
            answer = row[config["answer_field"]].strip()
        elif task_name == "mmlu_pro":
            question = format_mmlu_question(row)
            raw_ans = row[config["answer_field"]].strip()
            answer = f"({raw_ans})" if not raw_ans.startswith("(") else raw_ans
        else:
            raise ValueError(f"Unknown task: {task_name}")

        examples.append({
            "question": question,
            "answer": answer,
            "raw": row,
        })

    return examples


def parse_model_answer(response: str, answer_type: str) -> str:
    """Extract the model's answer from its response text."""
    # Try to find "ANSWER: X" pattern first
    match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
    else:
        # Fallback: take the last line or last parenthesized letter
        ans = response.strip().split("\n")[-1].strip()

    if answer_type == "numeric":
        # Extract number
        nums = re.findall(r"[\d,]+\.?\d*", ans)
        if nums:
            return nums[-1].replace(",", "").rstrip(".")
        return ans
    elif answer_type == "multiple_choice":
        # Extract letter in parentheses
        letters = re.findall(r"\(([A-Z])\)", ans)
        if letters:
            return f"({letters[-1]})"
        # Try bare letter
        letters = re.findall(r"\b([A-Z])\b", ans)
        if letters:
            return f"({letters[-1]})"
        return ans

    return ans


def check_answer(predicted: str, gold: str, answer_type: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred = predicted.strip().lower()
    gold = gold.strip().lower()

    if answer_type == "numeric":
        try:
            return float(pred.replace(",", "")) == float(gold.replace(",", ""))
        except ValueError:
            return pred == gold

    return pred == gold
