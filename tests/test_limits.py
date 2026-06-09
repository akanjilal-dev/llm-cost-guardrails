"""
tests/test_limits.py
====================
Rate limiting and the kill switch are tested against a controllable clock, so
the time-based behaviour is deterministic and there is no real sleeping.
"""

import pytest

from guardrails.limits import (
    KillSwitch,
    KillSwitchTripped,
    RateLimiter,
    RateLimitExceeded,
    TokenBucket,
)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def test_token_bucket_drains_and_refills():
    clock = Clock()
    b = TokenBucket(capacity=10, refill_per_sec=1, clock=clock)
    assert all(b.take() for _ in range(10))   # drains the bucket
    assert not b.take()                        # empty
    clock.advance(5)
    assert sum(b.take() for _ in range(10)) == 5   # 5 seconds -> 5 tokens back


def test_request_rate_limit():
    clock = Clock()
    rl = RateLimiter(requests_per_min=3, tokens_per_min=10_000, clock=clock)
    for _ in range(3):
        rl.check("t1", tokens=100)
    with pytest.raises(RateLimitExceeded):
        rl.check("t1", tokens=100)


def test_token_rate_limit():
    clock = Clock()
    rl = RateLimiter(requests_per_min=1000, tokens_per_min=1000, clock=clock)
    rl.check("t1", tokens=800)
    with pytest.raises(RateLimitExceeded):
        rl.check("t1", tokens=300)             # 800 + 300 > 1000


def test_rejected_call_does_not_consume_request_allowance():
    clock = Clock()
    rl = RateLimiter(requests_per_min=1000, tokens_per_min=1000, clock=clock)
    with pytest.raises(RateLimitExceeded):
        rl.check("t1", tokens=5000)            # over token budget -> rejected
    # The request bucket must NOT have been charged for the rejected call.
    for _ in range(5):
        rl.check("t1", tokens=10)


def test_limits_are_per_tenant():
    clock = Clock()
    rl = RateLimiter(requests_per_min=1, tokens_per_min=10_000, clock=clock)
    rl.check("t1", tokens=100)
    rl.check("t2", tokens=100)                 # different tenant, own bucket


def test_kill_switch_trips_and_fails_closed():
    clock = Clock()
    ks = KillSwitch(limit_usd=1.0, window_sec=60, clock=clock)
    ks.check()                                 # fine
    ks.record(0.60)
    ks.record(0.60)                            # window spend 1.20 > 1.0 -> trip
    assert ks.tripped
    with pytest.raises(KillSwitchTripped):
        ks.check()


def test_kill_switch_window_rolls_off():
    clock = Clock()
    ks = KillSwitch(limit_usd=1.0, window_sec=60, clock=clock)
    ks.record(0.80)
    clock.advance(61)                          # the spend ages out of the window
    assert ks.window_spend == 0.0
    ks.record(0.80)                            # fresh window, still under limit
    assert not ks.tripped


def test_kill_switch_reset():
    ks = KillSwitch(limit_usd=1.0, window_sec=60, clock=Clock())
    ks.record(2.0)
    assert ks.tripped
    ks.reset()
    assert not ks.tripped
    ks.check()
