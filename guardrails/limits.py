"""
guardrails/limits.py
====================
Two rate controls and one circuit breaker.

  * TokenBucket   -- the classic algorithm: a bucket refills at a steady rate
                     and each request draws from it. Smooths bursts without a
                     fixed per-window cliff.
  * RateLimiter   -- per-tenant request-per-minute and token-per-minute buckets,
                     mirroring how providers themselves meter you.
  * KillSwitch    -- a spend circuit breaker. If total spend in a rolling window
                     crosses a ceiling, it trips and every call is refused until
                     a human resets it. This is the backstop for a runaway agent
                     loop that slips past per-tenant budgets.

Time is injected (a `clock` callable returning seconds) so behaviour is
deterministic under test and the same code runs in production against a real
monotonic clock.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


class RateLimitExceeded(Exception):
    """Raised when a tenant exceeds its request or token rate."""


class KillSwitchTripped(Exception):
    """Raised when the global spend circuit breaker has tripped."""


@dataclass
class TokenBucket:
    capacity: float
    refill_per_sec: float
    clock: Callable[[], float] = time.monotonic
    _tokens: float = field(default=None, init=False)  # type: ignore[assignment]
    _last: float = field(default=None, init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last = self.clock()

    def _refill(self) -> None:
        now = self.clock()
        elapsed = max(now - self._last, 0.0)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
        self._last = now

    def take(self, amount: float = 1.0) -> bool:
        """Draw `amount` from the bucket. Returns False if there isn't enough."""
        self._refill()
        if self._tokens + 1e-9 >= amount:
            self._tokens -= amount
            return True
        return False


@dataclass
class RateLimiter:
    """Per-tenant requests-per-minute and tokens-per-minute limits."""

    requests_per_min: float = 60
    tokens_per_min: float = 100_000
    clock: Callable[[], float] = time.monotonic
    _req: dict[str, TokenBucket] = field(default_factory=dict)
    _tok: dict[str, TokenBucket] = field(default_factory=dict)

    def _bucket(self, store: dict[str, TokenBucket], tenant: str, per_min: float) -> TokenBucket:
        if tenant not in store:
            store[tenant] = TokenBucket(capacity=per_min, refill_per_sec=per_min / 60.0, clock=self.clock)
        return store[tenant]

    def check(self, tenant: str, tokens: int) -> None:
        """Admit one request of `tokens`, or raise RateLimitExceeded.

        Both buckets must admit the call; if either is short, neither is charged,
        so a rejected call doesn't silently consume the other allowance.
        """
        req = self._bucket(self._req, tenant, self.requests_per_min)
        tok = self._bucket(self._tok, tenant, self.tokens_per_min)
        # Peek without mutating: refill, then compare, then commit only if both pass.
        req._refill()
        tok._refill()
        if req._tokens + 1e-9 < 1.0:
            raise RateLimitExceeded(
                f"tenant {tenant!r} over request rate ({self.requests_per_min}/min)"
            )
        if tok._tokens + 1e-9 < tokens:
            raise RateLimitExceeded(
                f"tenant {tenant!r} over token rate: needs {tokens}, "
                f"{int(tok._tokens)} left this window ({self.tokens_per_min}/min)"
            )
        req._tokens -= 1.0
        tok._tokens -= tokens


@dataclass
class KillSwitch:
    """A rolling-window spend circuit breaker. Fails closed once tripped."""

    limit_usd: float
    window_sec: float = 60.0
    clock: Callable[[], float] = time.monotonic
    tripped: bool = False
    _events: deque = field(default_factory=deque)  # (timestamp, cost)
    _window_spend: float = 0.0

    def _prune(self) -> None:
        cutoff = self.clock() - self.window_sec
        while self._events and self._events[0][0] < cutoff:
            _, cost = self._events.popleft()
            self._window_spend = round(self._window_spend - cost, 6)

    def check(self) -> None:
        """Raise KillSwitchTripped if the breaker is open."""
        if self.tripped:
            raise KillSwitchTripped(
                f"spend circuit breaker tripped: >${self.limit_usd:.2f} in {self.window_sec:.0f}s. "
                f"Investigate, then call reset()."
            )

    def record(self, cost: float) -> None:
        """Record settled spend; trip if the rolling window crosses the ceiling."""
        self._prune()
        self._events.append((self.clock(), cost))
        self._window_spend = round(self._window_spend + cost, 6)
        if self._window_spend > self.limit_usd:
            self.tripped = True

    @property
    def window_spend(self) -> float:
        self._prune()
        return self._window_spend

    def reset(self) -> None:
        self.tripped = False
        self._events.clear()
        self._window_spend = 0.0
