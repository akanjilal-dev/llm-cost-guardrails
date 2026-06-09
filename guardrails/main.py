"""
guardrails/main.py
==================
A demo you can run offline: simulate a stream of LLM requests from several
tenants through the CostGuard and watch the controls fire. No API keys, no
network. The "LLM" is a deterministic mock that reports token usage so the cost
arithmetic is exact and the output is reproducible.

    python -m guardrails.main
"""

from __future__ import annotations

from guardrails.budgets import BudgetExceeded
from guardrails.guard import default_guard
from guardrails.limits import KillSwitchTripped, RateLimitExceeded
from guardrails.pricing import Usage, estimate_cost


class FakeClock:
    """A controllable clock so the demo is deterministic (no real sleeping)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def mock_llm(model: str, input_tokens: int, output_tokens: int):
    """Stand-in for a provider call: returns (text, Usage) deterministically."""
    def _call() -> tuple[str, Usage]:
        return ("...response...", Usage(input_tokens=input_tokens, output_tokens=output_tokens))
    return _call


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)


def main() -> None:
    clock = FakeClock()
    guard = default_guard(
        per_tenant_budget=0.50,        # 50 cents per tenant
        requests_per_min=5,            # small, so the demo can trip it
        tokens_per_min=10_000_000,     # generous; we demo the request limit below
        kill_switch_limit=2.00,        # global $2 / minute breaker
        kill_switch_window_sec=60.0,
        clock=clock,
    )
    guard.budgets.set_cap("acme", 0.20)        # acme is on a tighter plan
    guard.budgets.set_cap("globex", 5.00)      # globex is an enterprise tenant
    guard.budgets.set_cap("spike", 100.0)      # a runaway agent with deep pockets

    _rule("llm-cost-guardrails — every LLM call, metered and bounded")

    # A normal call settles and is attributed.
    r = guard.guarded_call(
        tenant="globex", model="claude-opus-4-8",
        input_tokens=2000, max_output_tokens=1000,
        call=mock_llm("claude-opus-4-8", 2000, 800),
    )
    print(f"  globex  opus-4.8   in=2000 out=800   -> ${r.cost_usd:.5f}  (allowed)")

    # Same prompt on a cheaper model — see the cost difference the router buys you.
    for model in ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"):
        c = estimate_cost(model, Usage(input_tokens=2000, output_tokens=800))
        print(f"    same call on {model:<20} would cost ${c:.5f}")

    _rule("Per-tenant budget: acme has a $0.20 cap")
    spent = 0.0
    for i in range(1, 8):
        try:
            r = guard.guarded_call(
                tenant="acme", model="claude-opus-4-8",
                input_tokens=3000, max_output_tokens=1500,
                call=mock_llm("claude-opus-4-8", 3000, 1200),
            )
            spent += r.cost_usd
            print(f"  call {i}: ${r.cost_usd:.5f}  (allowed, acme spent ${spent:.5f})")
        except BudgetExceeded as e:
            print(f"  call {i}: REFUSED — {e}")
            break

    _rule("Rate limit: globex capped at 5 requests/min")
    allowed = 0
    for i in range(1, 9):
        try:
            guard.guarded_call(
                tenant="globex", model="claude-haiku-4-5",
                input_tokens=500, max_output_tokens=200,
                call=mock_llm("claude-haiku-4-5", 500, 150),
            )
            allowed += 1
        except RateLimitExceeded as e:
            print(f"  request {i}: REFUSED — {e}")
            break
    print(f"  {allowed} requests admitted before the limiter said no")
    clock.advance(60)  # a minute passes; the bucket refills
    guard.guarded_call(
        tenant="globex", model="claude-haiku-4-5",
        input_tokens=500, max_output_tokens=200,
        call=mock_llm("claude-haiku-4-5", 500, 150),
    )
    print("  after 60s the bucket refilled — next request admitted again")

    _rule("Kill switch: global $2.00 / minute spend breaker")
    big = mock_llm("claude-opus-4-8", 50_000, 50_000)  # ~$1.50 per call
    for i in range(1, 5):
        try:
            r = guard.guarded_call(
                tenant="spike", model="claude-opus-4-8",
                input_tokens=50_000, max_output_tokens=50_000, call=big,
            )
            print(f"  burst call {i}: ${r.cost_usd:.4f}  (window spend ${guard.kill_switch.window_spend:.4f})")
        except KillSwitchTripped as e:
            print(f"  burst call {i}: REFUSED — {e}")
            break

    _rule("Per-tenant cost attribution")
    for tenant, row in guard.budgets.report().items():
        print(f"  {tenant:<8} spent ${row['spent']:.5f} of ${row['cap']:.2f}  (${row['remaining']:.5f} left)")
    print("=" * 72)


if __name__ == "__main__":
    main()
