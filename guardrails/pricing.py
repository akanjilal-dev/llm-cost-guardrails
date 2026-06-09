"""
guardrails/pricing.py
=====================
Turn token usage into dollars. Every other control in this package decides
whether to allow a call by first asking this module what the call will cost.

Pricing is per-million-tokens, the unit every major provider publishes. The
table below carries published Anthropic Claude rates; cached input is billed at
a fraction of the base input rate (a cache read is far cheaper than reprocessing
the tokens, a cache write slightly more expensive than processing them once).

Rates drift. Treat this table as a starting point and re-verify against the
provider's pricing page before you rely on a number. The guardrails are exact
about *arithmetic*, not about the rate card.
"""

from __future__ import annotations

from dataclasses import dataclass

# A cache read costs about a tenth of the base input rate; a cache write about
# 1.25x (the five minute time to live). Published Anthropic multipliers.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


# Published Anthropic Claude rates (USD per million tokens). Verify before use.
PRICING: dict[str, ModelPrice] = {
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

# When a request names a model we do not recognise, price it as the most
# expensive known model rather than guessing low. A budget control that
# under-estimates an unknown model is worse than useless.
_FALLBACK = ModelPrice(5.00, 25.00)


class UnknownModelError(KeyError):
    """Raised when strict pricing is requested for an unrecognised model."""


def price_for(model: str, *, strict: bool = False) -> ModelPrice:
    price = PRICING.get(model)
    if price is None:
        if strict:
            raise UnknownModelError(model)
        return _FALLBACK
    return price


@dataclass(frozen=True)
class Usage:
    """Token counts for one LLM call, matching the shape providers report back."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def estimate_cost(model: str, usage: Usage, *, strict: bool = False) -> float:
    """Cost in USD for a given model and token usage."""
    price = price_for(model, strict=strict)
    base_in = price.input_per_million / _PER_MILLION
    base_out = price.output_per_million / _PER_MILLION
    return round(
        usage.input_tokens * base_in
        + usage.output_tokens * base_out
        + usage.cache_read_tokens * base_in * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * base_in * CACHE_WRITE_MULTIPLIER,
        6,
    )


def worst_case_cost(model: str, input_tokens: int, max_output_tokens: int, *, strict: bool = False) -> float:
    """The most a call can cost: full input plus the output ceiling.

    This is what budget reservation should charge against *before* the call,
    because the response length is not known until it returns.
    """
    return estimate_cost(
        model,
        Usage(input_tokens=input_tokens, output_tokens=max_output_tokens),
        strict=strict,
    )
