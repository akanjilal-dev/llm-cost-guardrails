"""
tests/test_pricing.py
=====================
Cost arithmetic must be exact — it is the number every other control trusts.
"""

import pytest

from guardrails.pricing import (
    Usage,
    UnknownModelError,
    estimate_cost,
    price_for,
    worst_case_cost,
)


def test_opus_input_output_cost():
    # 1M input @ $5 + 1M output @ $25 = $30
    cost = estimate_cost("claude-opus-4-8", Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == pytest.approx(30.0)


def test_sonnet_cheaper_than_opus():
    u = Usage(input_tokens=10_000, output_tokens=5_000)
    assert estimate_cost("claude-sonnet-4-6", u) < estimate_cost("claude-opus-4-8", u)


def test_cache_read_is_one_tenth_of_input():
    read = estimate_cost("claude-opus-4-8", Usage(cache_read_tokens=1_000_000))
    fresh = estimate_cost("claude-opus-4-8", Usage(input_tokens=1_000_000))
    assert read == pytest.approx(fresh * 0.10)


def test_cache_write_premium():
    write = estimate_cost("claude-opus-4-8", Usage(cache_write_tokens=1_000_000))
    fresh = estimate_cost("claude-opus-4-8", Usage(input_tokens=1_000_000))
    assert write == pytest.approx(fresh * 1.25)


def test_unknown_model_falls_back_to_most_expensive():
    # Non-strict: price as the priciest known model, never under-estimate.
    assert price_for("some-future-model").input_per_million == 5.00


def test_unknown_model_strict_raises():
    with pytest.raises(UnknownModelError):
        estimate_cost("some-future-model", Usage(input_tokens=10), strict=True)


def test_worst_case_uses_output_ceiling():
    wc = worst_case_cost("claude-opus-4-8", input_tokens=1000, max_output_tokens=4000)
    actual = estimate_cost("claude-opus-4-8", Usage(input_tokens=1000, output_tokens=500))
    assert wc > actual  # the hold is always >= the settled cost
