"""Output-token floor for reasoning models used as the OSWorld 2.0 evaluator (host side only).

Some tasks pass a small per-call ``max_tokens`` to the evaluator (for example 1500). A reasoning
model can spend that whole cap on reasoning and return no final JSON. For the evaluator models
listed in ``RV_EVAL_MIN_OUTPUT_MODELS`` (comma-separated, empty by default) the resolved cap is
raised to ``RV_EVAL_MIN_OUTPUT_TOKENS`` (default 16384). Rubrics, inputs, other options and higher
caps are unchanged, and the upstream checkout is not edited.
"""
from __future__ import annotations

import os

DEFAULT_FLOOR = 16384


def policy() -> dict:
    models = sorted({m.strip() for m in os.environ.get("RV_EVAL_MIN_OUTPUT_MODELS", "").split(",") if m.strip()})
    floor = int(os.environ.get("RV_EVAL_MIN_OUTPUT_TOKENS", str(DEFAULT_FLOOR)))
    if floor <= 0:
        raise ValueError("RV_EVAL_MIN_OUTPUT_TOKENS must be positive")
    return {"models": models, "min_output_tokens": floor}


def install() -> dict:
    """Wrap ``model_client._build_config`` (idempotent) and return the active policy."""
    from desktop_env.evaluators import model_client

    info = policy()
    models = frozenset(info["models"])
    original = getattr(model_client._build_config, "_rv_eval_original", model_client._build_config)

    def build(*args, **kwargs):
        config = original(*args, **kwargs)
        if config.model in models:
            config.max_tokens = max(config.max_tokens, info["min_output_tokens"])
        return config

    build._rv_eval_original = original
    model_client._build_config = build
    return info
