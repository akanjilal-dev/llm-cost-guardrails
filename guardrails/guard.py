"""
guardrails/guard.py
===================
The middleware. `CostGuard` wraps any LLM call and enforces, in order:

  1. the global spend circuit breaker (kill switch),
  2. the per-tenant rate limits (requests and tokens per minute),
  3. the per-tenant budget, reserving the worst-case cost before the call,

then runs the call, reconciles the budget to the *actual* cost, records the
spend against the breaker, and attributes it to the tenant. Every gate fails
closed: if a control says no, the underlying LLM is never called, so you do not
pay for a request you were not allowed to make.

The wrapped call is any callable returning `(result, Usage)`. That keeps this
provider-agnostic: point it at a real Anthropic or OpenAI client in production,
or at the deterministic mock in `main.py` to run the whole thing offline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from guardrails.budgets import BudgetLedger
from guardrails.limits import KillSwitch, RateLimiter
from guardrails.pricing import Usage, estimate_cost, worst_case_cost


@dataclass
class GuardedResult:
    result: object
    usage: Usage
    cost_usd: float
    tenant: str
    model: str


class CostGuard:
    def __init__(
        self,
        *,
        budgets: BudgetLedger,
        limiter: RateLimiter,
        kill_switch: KillSwitch,
        strict_pricing: bool = False,
    ):
        self.budgets = budgets
        self.limiter = limiter
        self.kill_switch = kill_switch
        self.strict_pricing = strict_pricing

    def guarded_call(
        self,
        *,
        tenant: str,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
        call: Callable[[], tuple[object, Usage]],
    ) -> GuardedResult:
        # 1. Global circuit breaker — refuse everything if spend ran away.
        self.kill_switch.check()

        # 2. Rate limits — count the worst-case token footprint of this call.
        self.limiter.check(tenant, input_tokens + max_output_tokens)

        # 3. Budget — reserve the worst case before we spend a cent.
        hold = worst_case_cost(model, input_tokens, max_output_tokens, strict=self.strict_pricing)
        self.budgets.reserve(tenant, hold)

        # 4. Execute. If the call itself fails, release the hold — no charge.
        try:
            result, usage = call()
        except Exception:
            self.budgets.release(tenant, hold)
            raise

        # 5. Reconcile to the real cost and attribute it.
        actual = estimate_cost(model, usage, strict=self.strict_pricing)
        self.budgets.reconcile(tenant, hold, actual)
        self.kill_switch.record(actual)

        return GuardedResult(result=result, usage=usage, cost_usd=actual, tenant=tenant, model=model)


def default_guard(
    *,
    per_tenant_budget: float,
    requests_per_min: float,
    tokens_per_min: float,
    kill_switch_limit: float,
    kill_switch_window_sec: float = 60.0,
    clock: Callable[[], float] = time.monotonic,
) -> CostGuard:
    """Build a CostGuard with sensible, fully-wired defaults."""
    return CostGuard(
        budgets=BudgetLedger(default_cap=per_tenant_budget),
        limiter=RateLimiter(
            requests_per_min=requests_per_min, tokens_per_min=tokens_per_min, clock=clock
        ),
        kill_switch=KillSwitch(
            limit_usd=kill_switch_limit, window_sec=kill_switch_window_sec, clock=clock
        ),
    )
