"""Sanity checks for prompt assembly and experimental design."""

import numpy as np
import pytest

from src.components import COMPONENT_KEYS, load_components, load_task_config
from src.prompts import assemble_prompt, flags_from_vector, build_few_shot_block
from src.design import (
    generate_pb_design,
    generate_loo_design,
    generate_full_factorial,
    design_to_configs,
    compute_main_effects,
)
from src.datasets import extract_gsm8k_answer, parse_model_answer, check_answer


class TestComponents:
    def test_load_components(self):
        components = load_components()
        assert len(components) == 6
        for key in COMPONENT_KEYS:
            assert key in components

    def test_component_keys_order(self):
        assert COMPONENT_KEYS == [
            "system_role", "persona", "few_shot",
            "cot_trigger", "output_format", "constraints",
        ]


class TestPromptAssembly:
    def test_all_present(self):
        flags = {k: True for k in COMPONENT_KEYS}
        config = load_task_config("gsm8k")
        prompt = assemble_prompt(flags, config, "What is 2+2?")
        assert "What is 2+2?" in prompt
        assert "expert AI assistant" in prompt  # system_role present

    def test_all_absent(self):
        flags = {k: False for k in COMPONENT_KEYS}
        config = load_task_config("gsm8k")
        prompt = assemble_prompt(flags, config, "What is 2+2?")
        assert "What is 2+2?" in prompt
        assert "expert AI assistant" not in prompt

    def test_flags_from_vector(self):
        vec = [1, -1, 1, -1, 1, -1]
        flags = flags_from_vector(vec)
        assert flags["system_role"] is True
        assert flags["persona"] is False
        assert flags["few_shot"] is True

    def test_few_shot_present(self):
        flags = {k: True for k in COMPONENT_KEYS}
        config = load_task_config("gsm8k")
        prompt = assemble_prompt(flags, config, "Test?")
        assert "Example 1:" in prompt

    def test_few_shot_absent(self):
        flags = {k: True for k in COMPONENT_KEYS}
        flags["few_shot"] = False
        config = load_task_config("gsm8k")
        prompt = assemble_prompt(flags, config, "Test?")
        assert "Example 1:" not in prompt


class TestExperimentalDesign:
    def test_pb_design_shape(self):
        design = generate_pb_design(6)
        assert design.shape[1] == 6
        assert design.shape[0] >= 8  # pyDOE2 returns 8 for 6 factors
        assert set(design.flatten()) == {-1, 1}

    def test_loo_design_shape(self):
        design = generate_loo_design(6)
        assert design.shape == (7, 6)
        assert design[0].tolist() == [1, 1, 1, 1, 1, 1]

    def test_loo_design_structure(self):
        design = generate_loo_design(6)
        for i in range(6):
            assert design[i + 1, i] == -1
            assert sum(design[i + 1]) == 4  # 5 present, 1 absent

    def test_full_factorial_shape(self):
        design = generate_full_factorial(6)
        assert design.shape == (64, 6)

    def test_design_to_configs(self):
        design = np.array([[1, -1, 1, -1, 1, -1]])
        configs = design_to_configs(design)
        assert len(configs) == 1
        assert configs[0]["system_role"] == True
        assert configs[0]["persona"] == False

    def test_main_effects_known(self):
        # Simple 2-factor full factorial
        design = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        # Factor A has effect +0.2, factor B has effect +0.1
        responses = np.array([0.65, 0.45, 0.55, 0.35])
        effects = compute_main_effects(design, responses)
        np.testing.assert_almost_equal(effects[0], 0.2)
        np.testing.assert_almost_equal(effects[1], 0.1)


class TestDataParsing:
    def test_gsm8k_answer_extraction(self):
        assert extract_gsm8k_answer("blah blah #### 42") == "42"
        assert extract_gsm8k_answer("#### 1,234") == "1234"

    def test_parse_numeric_answer(self):
        assert parse_model_answer("ANSWER: 42", "numeric") == "42"
        assert parse_model_answer("The answer is 42.", "numeric") == "42"

    def test_parse_mc_answer(self):
        assert parse_model_answer("ANSWER: (B)", "multiple_choice") == "(B)"
        assert parse_model_answer("I think the answer is (C).", "multiple_choice") == "(C)"

    def test_check_answer_numeric(self):
        assert check_answer("42", "42", "numeric")
        assert check_answer("42.0", "42", "numeric")
        assert not check_answer("43", "42", "numeric")

    def test_check_answer_mc(self):
        assert check_answer("(B)", "(B)", "multiple_choice")
        assert check_answer("(b)", "(B)", "multiple_choice")
        assert not check_answer("(A)", "(B)", "multiple_choice")
