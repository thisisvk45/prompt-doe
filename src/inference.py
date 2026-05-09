# MIT License - Copyright (c) 2026 Vikas Kumar
from __future__ import annotations
"""Unified inference layer for Ollama (Llama) and Anthropic (Claude) models."""

import json
import hashlib
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

from src.components import load_model_config

CACHE_DIR = Path(__file__).parent.parent / "results" / "cache"


def _cache_key(model_id: str, prompt: str) -> str:
    """Generate a deterministic cache key."""
    content = f"{model_id}||{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def _get_cached(model_id: str, prompt: str) -> str | None:
    """Check disk cache for a previous response."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_id, prompt)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data["response"]
    return None


def _set_cached(model_id: str, prompt: str, response: str):
    """Save response to disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_id, prompt)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({
        "model_id": model_id,
        "prompt_hash": key,
        "response": response,
    }))


def _call_ollama(model_id: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Call a local Ollama model."""
    import ollama
    response = ollama.chat(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return response["message"]["content"]


def _call_openai(model_id: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Call OpenAI API."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_openrouter(model_id: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Call OpenRouter API (OpenAI-compatible)."""
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run_single_inference(
    model_name: str,
    prompt: str,
    use_cache: bool = True,
) -> str:
    """Run inference for a single prompt.

    Args:
        model_name: key from models.yaml (e.g., 'llama_8b', 'claude_haiku').
        prompt: the fully assembled prompt string.
        use_cache: whether to check/save disk cache.

    Returns:
        Model response text.
    """
    config = load_model_config(model_name)
    model_id = config["model_id"]

    if use_cache:
        cached = _get_cached(model_id, prompt)
        if cached is not None:
            return cached

    provider = config["provider"]
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 512)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            if provider == "ollama":
                response = _call_ollama(model_id, prompt, temperature, max_tokens)
            elif provider == "openai":
                response = _call_openai(model_id, prompt, temperature, max_tokens)
            elif provider == "openrouter":
                response = _call_openrouter(model_id, prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Retry {attempt + 1}/{max_retries} after error: {e}. Waiting {wait}s...")
            time.sleep(wait)

    if use_cache:
        _set_cached(model_id, prompt, response)

    # Rate limiting for API providers
    rpm = config.get("requests_per_minute")
    if rpm:
        time.sleep(60.0 / rpm)

    return response


def run_inference(
    model_name: str,
    prompts: list[str],
    use_cache: bool = True,
    verbose: bool = True,
    max_workers: int = 10,
) -> list[str]:
    """Run inference on a batch of prompts with concurrency.

    Args:
        model_name: key from models.yaml.
        prompts: list of fully assembled prompt strings.
        use_cache: whether to use disk cache.
        verbose: print progress.
        max_workers: number of concurrent threads.

    Returns:
        List of model response texts (order preserved).
    """
    results = [None] * len(prompts)
    done_count = 0

    def _run(idx, prompt):
        return idx, run_single_inference(model_name, prompt, use_cache=use_cache)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, i, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            idx, resp = future.result()
            results[idx] = resp
            done_count += 1
            if verbose and done_count % 50 == 0:
                print(f"  [{model_name}] {done_count}/{len(prompts)}", flush=True)

    return results
