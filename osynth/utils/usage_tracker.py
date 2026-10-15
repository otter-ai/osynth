"""Track API usage (tokens, TTS characters, GPU hours, cost) across the pipeline.

Supports cumulative tracking across partial runs: when a step loads from disk
instead of regenerating, its prior usage is preserved from the previous
``usage.json`` so the cumulative totals always reflect the full pipeline cost.
"""
from __future__ import annotations

import json
import logging
import os
import time

from . import fileio
from .llm_client import LLMResponse, detect_provider
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Estimated per-million-token pricing {model: {input: $, output: $}}.
# These static estimates may differ from current provider billing.
LLM_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    # Google Gemini (prompts up to 200K tokens; output includes thinking tokens)
    "gemini-3.1-pro": {"input": 2.00, "output": 12.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    # Fallback
    "default": {"input": 3.00, "output": 15.00},
}

# GPU hourly rates (for local inference cost estimation)
GPU_HOURLY_RATES: dict[str, float] = {
    "A100": 2.21,
    "H100": 3.49,
    "default": 2.21,
}


@dataclass
class StepUsage:
    step_name: str
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cache_creation_tokens: int = 0
    llm_cache_read_tokens: int = 0
    llm_calls: int = 0
    tts_characters: int = 0
    tts_calls: int = 0
    gpu_seconds: float = 0.0  # wall-clock GPU time for local models
    gpu_type: str = ""

    @property
    def has_activity(self) -> bool:
        return self.llm_calls > 0 or self.tts_calls > 0 or self.gpu_seconds > 0

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_cache_creation_tokens": self.llm_cache_creation_tokens,
            "llm_cache_read_tokens": self.llm_cache_read_tokens,
            "llm_calls": self.llm_calls,
            "tts_characters": self.tts_characters,
            "tts_calls": self.tts_calls,
            "gpu_seconds": round(self.gpu_seconds, 2),
            "gpu_type": self.gpu_type,
        }


def _step_from_dict(data: dict) -> StepUsage:
    """Reconstruct a StepUsage from a serialised dict (handles old field names)."""
    return StepUsage(
        step_name=data.get("step_name", ""),
        llm_input_tokens=data.get("llm_input_tokens", 0) or data.get("claude_input_tokens", 0),
        llm_output_tokens=data.get("llm_output_tokens", 0) or data.get("claude_output_tokens", 0),
        llm_cache_creation_tokens=data.get("llm_cache_creation_tokens", 0) or data.get("claude_cache_creation_tokens", 0),
        llm_cache_read_tokens=data.get("llm_cache_read_tokens", 0) or data.get("claude_cache_read_tokens", 0),
        llm_calls=data.get("llm_calls", 0) or data.get("claude_calls", 0),
        tts_characters=data.get("tts_characters", 0),
        tts_calls=data.get("tts_calls", 0),
        gpu_seconds=data.get("gpu_seconds", 0.0),
        gpu_type=data.get("gpu_type", ""),
    )


def _sum_steps(steps: dict[str, StepUsage]) -> dict[str, int | float]:
    return {
        "llm_input_tokens": sum(s.llm_input_tokens for s in steps.values()),
        "llm_output_tokens": sum(s.llm_output_tokens for s in steps.values()),
        "llm_calls": sum(s.llm_calls for s in steps.values()),
        "tts_characters": sum(s.tts_characters for s in steps.values()),
        "tts_calls": sum(s.tts_calls for s in steps.values()),
        "gpu_seconds": sum(s.gpu_seconds for s in steps.values()),
    }


class UsageTracker:
    def __init__(self):
        self._steps: dict[str, StepUsage] = {}
        self._prior_steps: dict[str, StepUsage] = {}
        self._current_step: str | None = None
        self._llm_model: str = "default"
        self._tts_model: str = "default"
        self._gpu_type: str = "default"
        # Per-model token accumulator so mixed-model runs (e.g. Sonnet writer
        # + Gemini judge + Opus refiner) compute correct blended costs.
        self._per_model_tokens: dict[str, dict[str, int]] = {}

    @property
    def _llm_provider(self) -> str:
        p = detect_provider(self._llm_model)
        return {"openai": "OpenAI", "google": "Google"}.get(p, "Anthropic")

    def set_models(self, llm_model: str, tts_model: str = "default") -> None:
        self._llm_model = llm_model
        self._tts_model = tts_model

    def set_gpu_type(self, gpu_type: str) -> None:
        self._gpu_type = gpu_type

    # ------------------------------------------------------------------
    # Prior usage (loaded from a previous run's usage.json)
    # ------------------------------------------------------------------

    def load_prior(self, output_dir: str) -> None:
        """Load per-step usage from a previous run so cached steps keep their costs."""
        path = os.path.join(output_dir, "usage.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for step_name, step_data in data.get("per_step", {}).items():
                self._prior_steps[step_name] = _step_from_dict(step_data)
            logger.info(
                "Loaded prior usage for %d steps from %s",
                len(self._prior_steps),
                path,
            )
        except Exception as exc:
            logger.warning("Failed to load prior usage from %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Step tracking
    # ------------------------------------------------------------------

    def set_step(self, step_name: str) -> None:
        self._current_step = step_name
        if step_name not in self._steps:
            self._steps[step_name] = StepUsage(step_name=step_name, gpu_type=self._gpu_type)

    def _get_step(self) -> StepUsage:
        if self._current_step is None:
            self.set_step("unknown")
        return self._steps[self._current_step]  # type: ignore[index]

    def track_llm_response(self, response: LLMResponse) -> None:
        """Track token usage from a unified LLMResponse."""
        step = self._get_step()
        step.llm_calls += 1
        in_tok = getattr(response, "input_tokens", 0) or 0
        out_tok = getattr(response, "output_tokens", 0) or 0
        step.llm_input_tokens += in_tok
        step.llm_output_tokens += out_tok
        step.llm_cache_creation_tokens += getattr(response, "cache_creation_tokens", 0) or 0
        step.llm_cache_read_tokens += getattr(response, "cache_read_tokens", 0) or 0
        # Accumulate per-model so mixed-model runs price each model correctly.
        model_key = getattr(response, "model", "") or self._llm_model
        bucket = self._per_model_tokens.setdefault(model_key, {"input": 0, "output": 0})
        bucket["input"] += in_tok
        bucket["output"] += out_tok

    def track_claude_response(self, response) -> None:  # type: ignore[no-untyped-def]
        """Track usage from a raw Anthropic SDK response (backward compat)."""
        step = self._get_step()
        step.llm_calls += 1
        try:
            usage = response.usage
        except AttributeError:
            return
        if usage is None:
            return
        step.llm_input_tokens += usage.input_tokens
        step.llm_output_tokens += usage.output_tokens
        step.llm_cache_creation_tokens += usage.cache_creation_input_tokens
        step.llm_cache_read_tokens += usage.cache_read_input_tokens

    def track_tts_call(self, text: str) -> None:
        step = self._get_step()
        step.tts_calls += 1
        step.tts_characters += len(text)

    def track_gpu_seconds(self, seconds: float) -> None:
        step = self._get_step()
        step.gpu_seconds += seconds

    @contextmanager
    def gpu_timer(self):
        """Context manager that automatically tracks GPU wall-clock time.

        Usage::

            with tracker.gpu_timer():
                run_tts_inference(...)
        """
        start = time.monotonic()
        try:
            yield
        finally:
            self.track_gpu_seconds(time.monotonic() - start)

    # ------------------------------------------------------------------
    # Cumulative: merge current run with prior data
    # ------------------------------------------------------------------

    def _cumulative_steps(self) -> dict[str, StepUsage]:
        """Build per-step view: current data for steps that ran, prior for the rest."""
        merged: dict[str, StepUsage] = dict(self._prior_steps)
        for name, step in self._steps.items():
            if step.has_activity:
                merged[name] = step
        return merged

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _llm_cost_from_per_model(self) -> tuple[float, float]:
        """Compute blended input/output LLM costs using per-model token counts.

        Falls back to pricing the total against ``_llm_model`` when no
        per-model breakdown is available (e.g. for prior steps loaded from
        disk, which don't carry a per-model split).
        """
        if not self._per_model_tokens:
            return 0.0, 0.0
        total_in = total_out = 0.0
        for model, tokens in self._per_model_tokens.items():
            prices = LLM_PRICING.get(model, LLM_PRICING["default"])
            total_in += tokens["input"] / 1_000_000 * prices["input"]
            total_out += tokens["output"] / 1_000_000 * prices["output"]
        return total_in, total_out

    def _cost_for(self, totals: dict[str, int | float]) -> dict[str, float]:
        # Use per-model breakdown when present for accurate blended pricing;
        # fall back to the global model rate (covers prior-step aggregates).
        blended_in, blended_out = self._llm_cost_from_per_model()
        if blended_in == 0.0 and blended_out == 0.0:
            prices = LLM_PRICING.get(self._llm_model, LLM_PRICING["default"])
            blended_in = totals["llm_input_tokens"] / 1_000_000 * prices["input"]
            blended_out = totals["llm_output_tokens"] / 1_000_000 * prices["output"]
        gpu_rate = GPU_HOURLY_RATES.get(self._gpu_type, GPU_HOURLY_RATES["default"])
        gpu_hrs = totals["gpu_seconds"] / 3600.0
        gpu_cost = gpu_hrs * gpu_rate
        return {
            "llm_input_cost": round(blended_in, 4),
            "llm_output_cost": round(blended_out, 4),
            "llm_total_cost": round(blended_in + blended_out, 4),
            "gpu_cost": round(gpu_cost, 4),
            "gpu_hours": round(gpu_hrs, 4),
            "total_cost": round(blended_in + blended_out + gpu_cost, 4),
        }

    # Current-run aggregation

    @property
    def total_llm_input_tokens(self) -> int:
        return sum(s.llm_input_tokens for s in self._steps.values())

    @property
    def total_llm_output_tokens(self) -> int:
        return sum(s.llm_output_tokens for s in self._steps.values())

    @property
    def total_llm_calls(self) -> int:
        return sum(s.llm_calls for s in self._steps.values())

    @property
    def total_tts_characters(self) -> int:
        return sum(s.tts_characters for s in self._steps.values())

    @property
    def total_tts_calls(self) -> int:
        return sum(s.tts_calls for s in self._steps.values())

    @property
    def total_gpu_seconds(self) -> float:
        return sum(s.gpu_seconds for s in self._steps.values())

    @property
    def total_gpu_hours(self) -> float:
        return self.total_gpu_seconds / 3600.0

    def estimate_cost(self) -> dict[str, float]:
        """Cost for the current run only."""
        return self._cost_for(_sum_steps(self._steps))

    def estimate_cumulative_cost(self) -> dict[str, float]:
        """Cost across all steps (current run + prior cached steps)."""
        return self._cost_for(_sum_steps(self._cumulative_steps()))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        run_totals = _sum_steps(self._steps)
        run_costs = self._cost_for(run_totals)

        cum_steps = self._cumulative_steps()
        cum_totals = _sum_steps(cum_steps)
        cum_costs = self._cost_for(cum_totals)

        has_prior = bool(self._prior_steps)
        prices = LLM_PRICING.get(self._llm_model, LLM_PRICING["default"])

        model_label = self._llm_model
        if len(self._per_model_tokens) > 1:
            model_label = ", ".join(sorted(self._per_model_tokens))

        lines = [
            "=" * 60,
            "USAGE SUMMARY",
            "=" * 60,
            "",
            f"LLM API ({self._llm_provider}: {model_label}):",
            f"  Pricing:             ${prices['input']:.2f} / $"
            f"{prices['output']:.2f} per 1M tokens (in/out)",
        ]

        if len(self._per_model_tokens) > 1:
            lines.append("  Per-model tokens:")
            for m, toks in sorted(self._per_model_tokens.items()):
                m_prices = LLM_PRICING.get(m, LLM_PRICING["default"])
                m_cost = (
                    toks["input"] / 1_000_000 * m_prices["input"]
                    + toks["output"] / 1_000_000 * m_prices["output"]
                )
                lines.append(
                    f"    {m}: {toks['input']:,} in / {toks['output']:,} out"
                    f"  (${m_cost:.4f})"
                )


        if has_prior:
            lines += [
                "",
                "  This run:",
                f"    Total calls:       {run_totals['llm_calls']:,}",
                f"    Input tokens:      {run_totals['llm_input_tokens']:,}",
                f"    Output tokens:     {run_totals['llm_output_tokens']:,}",
                f"    LLM cost:          ${run_costs['llm_total_cost']:.4f}",
                "",
                "  Cumulative (all steps):",
                f"    Total calls:       {cum_totals['llm_calls']:,}",
                f"    Input tokens:      {cum_totals['llm_input_tokens']:,}",
                f"    Output tokens:     {cum_totals['llm_output_tokens']:,}",
                f"    LLM cost:          ${cum_costs['llm_total_cost']:.4f}",
            ]
        else:
            lines += [
                f"  Total calls:         {run_totals['llm_calls']:,}",
                f"  Input tokens:        {run_totals['llm_input_tokens']:,}",
                f"  Output tokens:       {run_totals['llm_output_tokens']:,}",
                f"  Input cost:          ${run_costs['llm_input_cost']:.4f}",
                f"  Output cost:         ${run_costs['llm_output_cost']:.4f}",
                f"  LLM total:           ${run_costs['llm_total_cost']:.4f}",
            ]

        gpu_hrs_run = run_totals["gpu_seconds"] / 3600.0
        gpu_hrs_cum = cum_totals["gpu_seconds"] / 3600.0

        lines += ["", f"GPU ({self._gpu_type}):"]
        if has_prior:
            lines += [
                f"  This run:            {run_totals['tts_calls']:,} TTS calls, "
                f"{gpu_hrs_run:.4f} GPU-hrs, ${run_costs['gpu_cost']:.4f}",
                f"  Cumulative:          {cum_totals['tts_calls']:,} TTS calls, "
                f"{gpu_hrs_cum:.4f} GPU-hrs, ${cum_costs['gpu_cost']:.4f}",
            ]
        else:
            lines += [
                f"  Total TTS calls:     {run_totals['tts_calls']:,}",
                f"  Total TTS chars:     {run_totals['tts_characters']:,}",
                f"  GPU hours:           {gpu_hrs_run:.4f}",
                f"  GPU cost:            ${run_costs['gpu_cost']:.4f}",
            ]

        lines.append("")
        if has_prior:
            lines += [
                f"THIS RUN COST:         ${run_costs['total_cost']:.4f}",
                f"CUMULATIVE COST:       ${cum_costs['total_cost']:.4f}",
            ]
        else:
            lines.append(f"TOTAL ESTIMATED COST:  ${run_costs['total_cost']:.4f}")

        lines += ["", "Per-step breakdown:"]
        for step_name, step in sorted(cum_steps.items()):
            gpu_hrs = step.gpu_seconds / 3600.0
            cached = not self._steps.get(step_name, StepUsage(step_name="")).has_activity
            tag = " (cached)" if cached and has_prior else ""
            lines.append(
                f"  {step_name}{tag}: "
                f"{step.llm_calls} LLM calls "
                f"({step.llm_input_tokens:,} in / {step.llm_output_tokens:,} out), "
                f"{step.tts_calls} TTS calls ({step.tts_characters:,} chars), "
                f"{gpu_hrs:.3f} GPU-hrs"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        cum_steps = self._cumulative_steps()
        cum_totals = _sum_steps(cum_steps)
        cum_costs = self._cost_for(cum_totals)

        run_totals = _sum_steps(self._steps)
        run_costs = self._cost_for(run_totals)

        return {
            "llm_model": self._llm_model,
            "llm_provider": self._llm_provider,
            "tts_model": self._tts_model,
            "gpu_type": self._gpu_type,
            "totals": {
                "llm_input_tokens": cum_totals["llm_input_tokens"],
                "llm_output_tokens": cum_totals["llm_output_tokens"],
                "llm_calls": cum_totals["llm_calls"],
                "tts_characters": cum_totals["tts_characters"],
                "tts_calls": cum_totals["tts_calls"],
                "gpu_seconds": round(cum_totals["gpu_seconds"], 2),
                "gpu_hours": round(cum_totals["gpu_seconds"] / 3600.0, 4),
            },
            "costs": cum_costs,
            "this_run": {
                "totals": {
                    "llm_input_tokens": run_totals["llm_input_tokens"],
                    "llm_output_tokens": run_totals["llm_output_tokens"],
                    "llm_calls": run_totals["llm_calls"],
                    "tts_characters": run_totals["tts_characters"],
                    "tts_calls": run_totals["tts_calls"],
                    "gpu_seconds": round(run_totals["gpu_seconds"], 2),
                    "gpu_hours": round(run_totals["gpu_seconds"] / 3600.0, 4),
                },
                "costs": run_costs,
            },
            "per_step": {name: step.to_dict() for name, step in sorted(cum_steps.items())},
        }

    def save(self, output_dir: str) -> None:
        fileio.makedirs(output_dir)
        fileio.write_json(os.path.join(output_dir, "usage.json"), self.to_dict())
        fileio.write_text(os.path.join(output_dir, "usage.txt"), self.summary())
        logger.info("Usage report saved to %s", output_dir)


# Global instance
tracker = UsageTracker()
