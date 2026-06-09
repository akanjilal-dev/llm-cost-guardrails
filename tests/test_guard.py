"""
tests/test_guard.py
===================
End-to-end: the middleware enforces every gate, fails closed, and never charges
for a call the underlying model never made.
"""

import pytest

from guardrails.budgets import BudgetExceeded
from guardrails.guard import default_guard
from guardrails.limits import KillSwitchTripped, RateLimitExceeded
from guardrails.pricing import Usage


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def call_with(input_tokens, output_tokens):
    def _c():
        return ("ok", Usage(input_tokens=input_tokens, output_tokens=output_tokens))
    return _c


def make_guard(clock, **overrides):
    kwargs = dict(
        per_tenant_budget=1.0,
        requests_per_min=100,
        tokens_per_min=10_000_000,
        kill_switch_limit=100.0,
        clock=clock,
    )
    kwargs.update(overrides)
    return default_guard(**kwargs)


def test_allowed_call_settles_actual_cost():
    g = make_guard(Clock())
    r = g.guarded_call(
        tenant="t1", model="claude-haiku-4-5",
        input_tokens=1000, max_output_tokens=1000,
        call=call_with(1000, 200),
    )
    assert r.result == "ok"
    assert r.cost_usd > 0
    # Budget reflects actual (out=200), not the worst-case hold (out=1000).
    worst = g.budgets  # cost of 200 output < cost of 1000 output
    assert g.budgets.spent("t1") == pytest.approx(r.cost_usd)


def test_budget_blocks_before_calling_model():
    g = make_guard(Clock(), per_tenant_budget=0.0001)
    called = {"n": 0}

    def expensive():
        called["n"] += 1
        return ("ok", Usage(input_tokens=100, output_tokens=100))

    with pytest.raises(BudgetExceeded):
        g.guarded_call(
            tenant="t1", model="claude-opus-4-8",
            input_tokens=100_000, max_output_tokens=100_000, call=expensive,
        )
    assert called["n"] == 0          # the model was never invoked


def test_rate_limit_blocks_call():
    g = make_guard(Clock(), requests_per_min=1)
    g.guarded_call(tenant="t1", model="claude-haiku-4-5",
                   input_tokens=10, max_output_tokens=10, call=call_with(10, 5))
    with pytest.raises(RateLimitExceeded):
        g.guarded_call(tenant="t1", model="claude-haiku-4-5",
                       input_tokens=10, max_output_tokens=10, call=call_with(10, 5))


def test_kill_switch_blocks_after_runaway():
    g = make_guard(Clock(), kill_switch_limit=0.01)
    # First call settles and pushes window spend over the tiny limit.
    g.guarded_call(tenant="t1", model="claude-opus-4-8",
                   input_tokens=10_000, max_output_tokens=2000, call=call_with(10_000, 2000))
    with pytest.raises(KillSwitchTripped):
        g.guarded_call(tenant="t2", model="claude-haiku-4-5",
                       input_tokens=10, max_output_tokens=10, call=call_with(10, 5))


def test_failed_call_releases_hold_no_charge():
    g = make_guard(Clock())

    def boom():
        raise RuntimeError("provider 500")

    with pytest.raises(RuntimeError):
        g.guarded_call(tenant="t1", model="claude-opus-4-8",
                       input_tokens=1000, max_output_tokens=1000, call=boom)
    # No charge, and the hold was released so the full budget is available again.
    assert g.budgets.spent("t1") == 0.0
    assert g.budgets.available("t1") == pytest.approx(1.0)
